package store_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"batchseal/internal/store"

	"github.com/jackc/pgx/v5/pgxpool"
)

// These tests require a real PostgreSQL reachable via TEST_DATABASE_URL
// (default matches the compose service). They skip when it is unavailable,
// e.g. when run without a database present.
func testURL() string {
	if v := os.Getenv("TEST_DATABASE_URL"); v != "" {
		return v
	}
	return "postgres://postgres:postgres@localhost:5432/batchseal?sslmode=disable"
}

var (
	basePoolOnce sync.Once
	basePool     *pgxpool.Pool
	basePoolErr  error
)

func ensureBasePool() bool {
	basePoolOnce.Do(func() {
		c, err := pgxpool.New(context.Background(), testURL())
		if err == nil {
			err = c.Ping(context.Background())
		}
		basePool, basePoolErr = c, err
	})
	return basePoolErr == nil
}

// newStore points every test at an isolated schema inside the shared
// database. The returned function opens another store on the same schema,
// modelling a second API process arbitrating through the same database.
func newStore(ctx context.Context, t *testing.T) (s *store.Store, peer func() *store.Store) {
	t.Helper()

	if !ensureBasePool() {
		t.Skipf("real PostgreSQL not available at %q: %v", testURL(), basePoolErr)
	}

	schema := fmt.Sprintf("t_%d", time.Now().UnixNano())
	if _, err := basePool.Exec(ctx, "CREATE SCHEMA "+schema); err != nil {
		t.Fatalf("create schema: %v", err)
	}
	t.Cleanup(func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = basePool.Exec(cleanupCtx, "DROP SCHEMA "+schema+" CASCADE")
	})
	testSchemas.Store(t.Name(), schema)

	open := func() *store.Store {
		st, err := store.New(ctx, schemaURL(schema))
		if err != nil {
			t.Fatalf("open store: %v", err)
		}
		return st
	}
	return open(), open
}

func schemaURL(schema string) string {
	sep := "?"
	if containsQ(testURL()) {
		sep = "&"
	}
	return testURL() + sep + "search_path=" + schema
}

func containsQ(u string) bool {
	for i := 0; i < len(u); i++ {
		if u[i] == '?' {
			return true
		}
	}
	return false
}

// mustBatch creates a batch or fails the test.
func mustBatch(ctx context.Context, t *testing.T, s *store.Store, expected int) *store.Batch {
	t.Helper()
	b, err := s.CreateBatch(ctx, expected)
	if err != nil {
		t.Fatalf("create batch: %v", err)
	}
	return b
}

// testSchemas records the isolated schema newStore creates per test, so tests
// that stage interleavings on raw base-pool connections (search_path public
// by default) can qualify table names explicitly.
var testSchemas sync.Map // t.Name() -> schema

// testSchema returns the isolated schema created for the current test.
func testSchema(t *testing.T) string {
	t.Helper()
	v, ok := testSchemas.Load(t.Name())
	if !ok {
		t.Fatalf("no test schema recorded for %s", t.Name())
	}
	return v.(string)
}

// TestConcurrentFirstBootMigrate points many brand-new stores at one fresh
// schema at the same time, modelling every API instance racing to apply the
// schema on an empty database. Exactly one migration must win the advisory
// lock; the rest must complete without a catalog duplicate-key error.
func TestConcurrentFirstBootMigrate(t *testing.T) {
	ctx := context.Background()
	if !ensureBasePool() {
		t.Skipf("real PostgreSQL not available at %q: %v", testURL(), basePoolErr)
	}
	schema := fmt.Sprintf("mig_%d", time.Now().UnixNano())
	if _, err := basePool.Exec(ctx, "CREATE SCHEMA "+schema); err != nil {
		t.Fatalf("create empty schema: %v", err)
	}
	t.Cleanup(func() {
		cctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_, _ = basePool.Exec(cctx, "DROP SCHEMA IF EXISTS "+schema+" CASCADE")
	})

	const n = 8
	var wg sync.WaitGroup
	errs := make([]error, n)
	wg.Add(n)
	start := make(chan struct{})
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			<-start
			st, err := store.New(ctx, schemaURL(schema))
			if err != nil {
				errs[i] = err
				return
			}
			// Prove the migrated schema is usable, then release.
			_, err = st.CreateBatch(ctx, 1)
			if err != nil {
				errs[i] = err
			}
			st.Close()
		}(i)
	}
	close(start)
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("concurrent migrator %d failed: %v", i, err)
		}
	}
}

func TestCreateBatchValidatesRange(t *testing.T) {
	ctx := context.Background()
	s, _ := newStore(ctx, t)
	for _, n := range []int{0, -1, 10001} {
		if _, err := s.CreateBatch(ctx, n); err == nil {
			t.Fatalf("expected error for expectedChunks=%d", n)
		}
	}
}

func TestOutOfOrderThenSnapshot(t *testing.T) {
	ctx := context.Background()
	s, _ := newStore(ctx, t)
	b := mustBatch(ctx, t, s, 4)

	for _, seq := range []int{4, 2, 1} {
		res, err := s.SubmitChunk(ctx, b.ID, seq, []byte{byte('a' + seq)})
		if err != nil {
			t.Fatalf("submit %d: %v", seq, err)
		}
		if !res.Created {
			t.Fatalf("seq %d should be a first write", seq)
		}
	}

	snap, err := s.Snapshot(ctx, b.ID)
	if err != nil {
		t.Fatal(err)
	}
	if snap.Received != 3 || len(snap.Gaps) != 1 || snap.Gaps[0] != 3 {
		t.Fatalf("unexpected snapshot: %+v", snap)
	}
}

func TestRetransmissionIsIdempotent(t *testing.T) {
	ctx := context.Background()
	s, _ := newStore(ctx, t)
	b := mustBatch(ctx, t, s, 2)

	first, err := s.SubmitChunk(ctx, b.ID, 1, []byte("hello"))
	if err != nil {
		t.Fatal(err)
	}
	second, err := s.SubmitChunk(ctx, b.ID, 1, []byte("hello"))
	if err != nil {
		t.Fatal(err)
	}
	if !first.Created || second.Created {
		t.Fatalf("created flags wrong: first=%+v second=%+v", first, second)
	}
	if !first.ReceivedAt.Equal(second.ReceivedAt) {
		t.Fatalf("original confirmation time changed: %v vs %v", first.ReceivedAt, second.ReceivedAt)
	}

	// Same seq, different bytes -> conflict.
	if _, err := s.SubmitChunk(ctx, b.ID, 1, []byte("hellX")); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("want ErrConflict, got %v", err)
	}
}

func TestUTF8ByteIdentity(t *testing.T) {
	ctx := context.Background()
	s, _ := newStore(ctx, t)
	b := mustBatch(ctx, t, s, 1)

	// "é" as UTF-8 (0xC3 0xA9) versus Latin-1 (0xE9): same letter, different
	// bytes, so the retransmission must be a conflict.
	if _, err := s.SubmitChunk(ctx, b.ID, 1, []byte("caf\xc3\xa9")); err != nil {
		t.Fatal(err)
	}
	if _, err := s.SubmitChunk(ctx, b.ID, 1, []byte("caf\xe9")); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("different UTF-8 bytes should conflict, got %v", err)
	}
	// Identical bytes are the original acknowledgement.
	res, err := s.SubmitChunk(ctx, b.ID, 1, []byte("caf\xc3\xa9"))
	if err != nil || res.Created {
		t.Fatalf("identical retransmission wrong: res=%+v err=%v", res, err)
	}
}

func TestSeqRangeRejected(t *testing.T) {
	ctx := context.Background()
	s, _ := newStore(ctx, t)
	b := mustBatch(ctx, t, s, 2)

	for _, seq := range []int{0, -1, 3, 10000} {
		if _, err := s.SubmitChunk(ctx, b.ID, seq, []byte("x")); !errors.Is(err, store.ErrSeqRange) {
			t.Fatalf("seq=%d: want ErrSeqRange, got %v", seq, err)
		}
	}
}

func TestSealIncompleteKeepsOpen(t *testing.T) {
	ctx := context.Background()
	s, _ := newStore(ctx, t)
	b := mustBatch(ctx, t, s, 3)

	if _, err := s.SubmitChunk(ctx, b.ID, 1, []byte("a")); err != nil {
		t.Fatal(err)
	}
	snap, err := s.SealBatch(ctx, b.ID)
	if !errors.Is(err, store.ErrIncomplete) {
		t.Fatalf("want ErrIncomplete, got %v", err)
	}
	if snap.Status != store.StatusOpen {
		t.Fatalf("status changed to %s despite missing chunks", snap.Status)
	}
	if len(snap.Gaps) != 2 || snap.Gaps[0] != 2 || snap.Gaps[1] != 3 {
		t.Fatalf("gaps wrong: %v", snap.Gaps)
	}

	again, err := s.Snapshot(ctx, b.ID)
	if err != nil {
		t.Fatal(err)
	}
	if again.Status != store.StatusOpen {
		t.Fatalf("batch did not remain OPEN: %s", again.Status)
	}
}

func TestSealCompleteIsIdempotent(t *testing.T) {
	ctx := context.Background()
	s, _ := newStore(ctx, t)
	b := mustBatch(ctx, t, s, 2)

	if _, err := s.SubmitChunk(ctx, b.ID, 1, []byte("a")); err != nil {
		t.Fatal(err)
	}
	if _, err := s.SubmitChunk(ctx, b.ID, 2, []byte("b")); err != nil {
		t.Fatal(err)
	}
	sealed, err := s.SealBatch(ctx, b.ID)
	if err != nil {
		t.Fatalf("seal: %v", err)
	}
	if sealed.Status != store.StatusSealed || sealed.SealedAt == nil {
		t.Fatalf("seal result wrong: %+v", sealed)
	}

	// Repeated seal returns the existing result with the same sealedAt.
	again, err := s.SealBatch(ctx, b.ID)
	if err != nil {
		t.Fatalf("reseal: %v", err)
	}
	if again.Status != store.StatusSealed || !again.SealedAt.Equal(*sealed.SealedAt) {
		t.Fatalf("reseal altered result: %+v vs %+v", sealed, again)
	}

	// Sealed batch rejects changed content, allows identical retransmission.
	if _, err := s.SubmitChunk(ctx, b.ID, 1, []byte("changed")); !errors.Is(err, store.ErrSealed) {
		t.Fatalf("want ErrSealed, got %v", err)
	}
	if _, err := s.SubmitChunk(ctx, b.ID, 1, []byte("a")); err != nil {
		t.Fatalf("identical retransmission after seal rejected: %v", err)
	}
}

func TestDuplicateChunksDoNotInflateCount(t *testing.T) {
	ctx := context.Background()
	s, _ := newStore(ctx, t)
	b := mustBatch(ctx, t, s, 1)

	for range 5 {
		if _, err := s.SubmitChunk(ctx, b.ID, 1, []byte("x")); err != nil {
			t.Fatal(err)
		}
	}
	snap, err := s.Snapshot(ctx, b.ID)
	if err != nil {
		t.Fatal(err)
	}
	if snap.Received != 1 || len(snap.Gaps) != 0 {
		t.Fatalf("count inflated by duplicates: %+v", snap)
	}
}

// TestSealRacesFinalChunk fires the seal and the last missing chunk
// concurrently from independent pools (simulating two API processes). The
// batch can never end up SEALED with a gap.
func TestSealRacesFinalChunk(t *testing.T) {
	ctx := context.Background()
	s, newPeer := newStore(ctx, t)
	other := newPeer()
	defer other.Close()

	const rounds = 30
	start := make(chan struct{})
	for i := 0; i < rounds; i++ {
		b := mustBatch(ctx, t, s, 1)

		var wg sync.WaitGroup
		wg.Add(2)
		var sealErr error
		go func() {
			defer wg.Done()
			<-start
			_, sealErr = s.SealBatch(ctx, b.ID)
		}()
		go func() {
			defer wg.Done()
			<-start
			_, _ = other.SubmitChunk(ctx, b.ID, 1, []byte("final"))
		}()
		close(start)
		wg.Wait()

		snap, err := s.Snapshot(ctx, b.ID)
		if err != nil {
			t.Fatal(err)
		}
		switch snap.Status {
		case store.StatusOpen:
			if !errors.Is(sealErr, store.ErrIncomplete) {
				t.Fatalf("OPEN but seal error was %v", sealErr)
			}
			// OPEN is consistent whether the chunk is still missing or it
			// landed just after the seal verdict; SEALED-with-gaps is the
			// forbidden outcome.
			if snap.Received+len(snap.Gaps) != snap.ExpectedChunks {
				t.Fatalf("OPEN snapshot inconsistent: %+v", snap)
			}
			// If the chunk did land, another seal must now succeed atomically.
			if len(snap.Gaps) == 0 {
				again, err := s.SealBatch(ctx, b.ID)
				if err != nil || again.Status != store.StatusSealed {
					t.Fatalf("follow-up seal failed: snap=%+v err=%v", again, err)
				}
			}
		case store.StatusSealed:
			if len(snap.Gaps) != 0 || snap.Received != snap.ExpectedChunks {
				t.Fatalf("SEALED batch has gaps: %+v", snap)
			}
		default:
			t.Fatalf("unknown status %s", snap.Status)
		}
		start = make(chan struct{})
	}
}

// TestConflictingConcurrentSubmits hammers one seq with two payloads from two
// independent pools: exactly one payload must win, the other must see a
// conflict, and exactly one row must remain stored.
func TestConflictingConcurrentSubmits(t *testing.T) {
	ctx := context.Background()
	s, newPeer := newStore(ctx, t)
	other := newPeer()
	defer other.Close()

	const rounds = 30
	for range rounds {
		b := mustBatch(ctx, t, s, 1)
		var wg sync.WaitGroup
		wg.Add(2)
		var err1, err2 error
		go func() { defer wg.Done(); _, err1 = s.SubmitChunk(ctx, b.ID, 1, []byte("AAAA")) }()
		go func() { defer wg.Done(); _, err2 = other.SubmitChunk(ctx, b.ID, 1, []byte("BBBB")) }()
		wg.Wait()

		conflicts := 0
		for _, e := range []error{err1, err2} {
			if errors.Is(e, store.ErrConflict) {
				conflicts++
			} else if e != nil {
				t.Fatalf("unexpected submit error: %v", e)
			}
		}
		if conflicts != 1 {
			t.Fatalf("expected exactly 1 conflict, got %d (err1=%v err2=%v)", conflicts, err1, err2)
		}
		snap, err := s.Snapshot(ctx, b.ID)
		if err != nil {
			t.Fatal(err)
		}
		if snap.Received != 1 {
			t.Fatalf("expected 1 stored chunk, got %d", snap.Received)
		}
	}
}

// TestSealGroupSuccessAndRetryStable seals a complete group and confirms the
// response follows request order and a retry (even reversed) is an idempotent
// success returning the original sealedAt for every member.
func TestSealGroupSuccessAndRetryStable(t *testing.T) {
	ctx := context.Background()
	s, _ := newStore(ctx, t)

	a := mustBatch(ctx, t, s, 2)
	b := mustBatch(ctx, t, s, 3)
	for seq, p := range map[int]string{1: "a1", 2: "a2"} {
		if _, err := s.SubmitChunk(ctx, a.ID, seq, []byte(p)); err != nil {
			t.Fatal(err)
		}
	}
	for seq, p := range map[int]string{1: "b1", 2: "b2", 3: "b3"} {
		if _, err := s.SubmitChunk(ctx, b.ID, seq, []byte(p)); err != nil {
			t.Fatal(err)
		}
	}

	snaps, err := s.SealGroup(ctx, []string{a.ID, b.ID})
	if err != nil {
		t.Fatalf("seal group: %v", err)
	}
	if len(snaps) != 2 || snaps[0].ID != a.ID || snaps[1].ID != b.ID {
		t.Fatalf("snapshots not in request order: %+v", snaps)
	}
	for _, snap := range snaps {
		if snap.Status != store.StatusSealed || snap.SealedAt == nil ||
			snap.Received != snap.ExpectedChunks || len(snap.Gaps) != 0 {
			t.Fatalf("member not cleanly sealed: %+v", snap)
		}
	}
	sealedAtA, sealedAtB := *snaps[0].SealedAt, *snaps[1].SealedAt

	// Retry in reversed order: idempotent, original sealedAt preserved.
	again, err := s.SealGroup(ctx, []string{b.ID, a.ID})
	if err != nil {
		t.Fatalf("retry: %v", err)
	}
	if again[0].ID != b.ID || again[1].ID != a.ID {
		t.Fatalf("retry snapshots not in request order: %+v", again)
	}
	if !again[0].SealedAt.Equal(sealedAtB) || !again[1].SealedAt.Equal(sealedAtA) {
		t.Fatalf("retry changed sealedAt: %v / %v", again[0].SealedAt, again[1].SealedAt)
	}
}

// TestSealGroupIncompleteLeavesWholeGroupUnchanged proves atomicity: with A
// complete and B short of chunks, the group seal fails and A is NOT sealed
// either — no per-member SealBatch loop leaking partial commits.
func TestSealGroupIncompleteLeavesWholeGroupUnchanged(t *testing.T) {
	ctx := context.Background()
	s, _ := newStore(ctx, t)

	a := mustBatch(ctx, t, s, 2) // will be complete
	b := mustBatch(ctx, t, s, 3) // will miss seq 2
	for seq, p := range map[int]string{1: "a1", 2: "a2"} {
		if _, err := s.SubmitChunk(ctx, a.ID, seq, []byte(p)); err != nil {
			t.Fatal(err)
		}
	}
	for seq, p := range map[int]string{1: "b1", 3: "b3"} {
		if _, err := s.SubmitChunk(ctx, b.ID, seq, []byte(p)); err != nil {
			t.Fatal(err)
		}
	}

	snaps, err := s.SealGroup(ctx, []string{a.ID, b.ID})
	if !errors.Is(err, store.ErrIncomplete) {
		t.Fatalf("want ErrIncomplete, got %v", err)
	}
	if len(snaps) != 2 || snaps[0].ID != a.ID || snaps[1].ID != b.ID {
		t.Fatalf("error snapshots not in request order: %+v", snaps)
	}
	if snaps[1].Received != 2 || len(snaps[1].Gaps) != 1 || snaps[1].Gaps[0] != 2 {
		t.Fatalf("incomplete member snapshot wrong: %+v", snaps[1])
	}

	// Neither member changed: A must still be OPEN with no sealedAt.
	for _, id := range []string{a.ID, b.ID} {
		snap, err := s.Snapshot(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		if snap.Status != store.StatusOpen || snap.SealedAt != nil {
			t.Fatalf("member changed despite failed group seal: %+v", snap)
		}
	}

	// Completing B makes the retry seal the whole group.
	if _, err := s.SubmitChunk(ctx, b.ID, 2, []byte("b2")); err != nil {
		t.Fatal(err)
	}
	sealed, err := s.SealGroup(ctx, []string{a.ID, b.ID})
	if err != nil {
		t.Fatalf("retry after completing B: %v", err)
	}
	if sealed[0].Status != store.StatusSealed || sealed[1].Status != store.StatusSealed {
		t.Fatalf("group not sealed after completing B: %+v", sealed)
	}
}

// TestSealGroupUnknownIDHasNoSideEffects: a ghost id fails the whole request
// with ErrNotFound and the existing, complete member stays OPEN.
func TestSealGroupUnknownIDHasNoSideEffects(t *testing.T) {
	ctx := context.Background()
	s, _ := newStore(ctx, t)

	a := mustBatch(ctx, t, s, 1)
	if _, err := s.SubmitChunk(ctx, a.ID, 1, []byte("x")); err != nil {
		t.Fatal(err)
	}
	ghost := "ffffffffffffffffffffffffffffffff" // valid format, never created

	if _, err := s.SealGroup(ctx, []string{a.ID, ghost}); !errors.Is(err, store.ErrNotFound) {
		t.Fatalf("want ErrNotFound, got %v", err)
	}
	snap, err := s.Snapshot(ctx, a.ID)
	if err != nil {
		t.Fatal(err)
	}
	if snap.Status != store.StatusOpen || snap.SealedAt != nil {
		t.Fatalf("known member changed by unknown-id group: %+v", snap)
	}
}

// TestSealGroupKeepsExistingSealedAt: a member sealed individually before the
// group call is an idempotent success and keeps its original sealedAt while
// the remaining OPEN members seal.
func TestSealGroupKeepsExistingSealedAt(t *testing.T) {
	ctx := context.Background()
	s, _ := newStore(ctx, t)

	a := mustBatch(ctx, t, s, 1)
	b := mustBatch(ctx, t, s, 1)
	for _, id := range []string{a.ID, b.ID} {
		if _, err := s.SubmitChunk(ctx, id, 1, []byte("x")); err != nil {
			t.Fatal(err)
		}
	}
	sealedA, err := s.SealBatch(ctx, a.ID)
	if err != nil {
		t.Fatal(err)
	}

	snaps, err := s.SealGroup(ctx, []string{a.ID, b.ID})
	if err != nil {
		t.Fatalf("group with pre-sealed member: %v", err)
	}
	if !snaps[0].SealedAt.Equal(*sealedA.SealedAt) {
		t.Fatalf("group seal overwrote existing sealedAt: %v vs %v", snaps[0].SealedAt, sealedA.SealedAt)
	}
	if snaps[1].Status != store.StatusSealed || snaps[1].SealedAt == nil {
		t.Fatalf("open member not sealed by group: %+v", snaps[1])
	}
}

// TestSealGroupThreeWayRace fires the [A,B] group, the reversed [B,A] group
// and B's final chunk from three independent pools at the same time. The
// shared row-lock arbitration must serialise them without deadlock, and the
// only legal outcomes are the whole group SEALED (by exactly one transaction,
// so both members carry the same sealedAt) or no new seal at all.
func TestSealGroupThreeWayRace(t *testing.T) {
	ctx := context.Background()
	s, newPeer := newStore(ctx, t)
	p2 := newPeer()
	defer p2.Close()
	p3 := newPeer()
	defer p3.Close()

	const rounds = 25
	for i := 0; i < rounds; i++ {
		a := mustBatch(ctx, t, s, 1)
		b := mustBatch(ctx, t, s, 1)
		if _, err := s.SubmitChunk(ctx, a.ID, 1, []byte("a")); err != nil {
			t.Fatal(err)
		}

		opCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		var wg sync.WaitGroup
		wg.Add(3)
		var snapsAB, snapsBA []*store.Snapshot
		var errAB, errBA error
		go func() { defer wg.Done(); snapsAB, errAB = s.SealGroup(opCtx, []string{a.ID, b.ID}) }()
		go func() { defer wg.Done(); snapsBA, errBA = p2.SealGroup(opCtx, []string{b.ID, a.ID}) }()
		go func() { defer wg.Done(); _, _ = p3.SubmitChunk(opCtx, b.ID, 1, []byte("final")) }()

		done := make(chan struct{})
		go func() { wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(30 * time.Second):
			t.Fatalf("round %d: three-way seal race deadlocked", i)
		}
		cancel()

		for _, e := range []error{errAB, errBA} {
			if e != nil && !errors.Is(e, store.ErrIncomplete) {
				t.Fatalf("round %d: unexpected group error %v", i, e)
			}
		}

		snapA, err := s.Snapshot(ctx, a.ID)
		if err != nil {
			t.Fatal(err)
		}
		snapB, err := s.Snapshot(ctx, b.ID)
		if err != nil {
			t.Fatal(err)
		}
		sealedA := snapA.Status == store.StatusSealed
		sealedB := snapB.Status == store.StatusSealed
		if sealedA != sealedB {
			t.Fatalf("round %d: group partially sealed: A=%s B=%s", i, snapA.Status, snapB.Status)
		}
		// A 200 response must never contradict the database: every member it
		// reports has to be SEALED with a sealedAt equal to the stored one.
		assertGroupResponseConsistent := func(label string, ordered []*store.Snapshot, e error) {
			t.Helper()
			if e != nil {
				return // INCOMPLETE responses carry no success snapshot
			}
			for _, got := range ordered {
				var stored *store.Snapshot
				switch got.ID {
				case a.ID:
					stored = snapA
				case b.ID:
					stored = snapB
				}
				if got.Status != stored.Status ||
					(got.SealedAt == nil) != (stored.SealedAt == nil) ||
					(got.SealedAt != nil && !got.SealedAt.Equal(*stored.SealedAt)) {
					t.Fatalf("round %d: %s 200 snapshot %+v disagrees with database %+v",
						i, label, got, stored)
				}
			}
		}
		assertGroupResponseConsistent("[A,B]", snapsAB, errAB)
		assertGroupResponseConsistent("[B,A]", snapsBA, errBA)
		// An INCOMPLETE response must carry the adjudicated verdict: some
		// member still OPEN with a non-empty gap list. A failure snapshot in
		// which every member already looks complete (or SEALED) contradicts
		// the 409 and hides what was missing.
		assertIncompleteNamesGap := func(label string, ordered []*store.Snapshot, e error) {
			t.Helper()
			if !errors.Is(e, store.ErrIncomplete) {
				return
			}
			for _, got := range ordered {
				if got.Status == store.StatusOpen && len(got.Gaps) > 0 {
					return
				}
			}
			t.Fatalf("round %d: %s INCOMPLETE names no incomplete member: %+v", i, label, ordered)
		}
		assertIncompleteNamesGap("[A,B]", snapsAB, errAB)
		assertIncompleteNamesGap("[B,A]", snapsBA, errBA)
		if sealedA {
			if len(snapA.Gaps) != 0 || len(snapB.Gaps) != 0 {
				t.Fatalf("round %d: SEALED with gaps: %+v %+v", i, snapA, snapB)
			}
			if !snapA.SealedAt.Equal(*snapB.SealedAt) {
				t.Fatalf("round %d: members sealed by different transactions: %v vs %v",
					i, snapA.SealedAt, snapB.SealedAt)
			}
		} else {
			// No new seal happened: both group calls must have reported
			// INCOMPLETE and nothing may have transitioned.
			if !errors.Is(errAB, store.ErrIncomplete) || !errors.Is(errBA, store.ErrIncomplete) {
				t.Fatalf("round %d: unsealed but errors were %v / %v", i, errAB, errBA)
			}
		}
	}
}

// TestSealGroupVsSingleSealRace: a group seal and a single-batch seal share
// the same row locks, so they serialise — the loser observes the winner's
// verdict idempotently and no sealedAt is ever overwritten.
func TestSealGroupVsSingleSealRace(t *testing.T) {
	ctx := context.Background()
	s, newPeer := newStore(ctx, t)
	other := newPeer()
	defer other.Close()

	const rounds = 25
	for i := 0; i < rounds; i++ {
		a := mustBatch(ctx, t, s, 1)
		b := mustBatch(ctx, t, s, 1)
		for _, id := range []string{a.ID, b.ID} {
			if _, err := s.SubmitChunk(ctx, id, 1, []byte("x")); err != nil {
				t.Fatal(err)
			}
		}

		var wg sync.WaitGroup
		wg.Add(2)
		var groupSnaps []*store.Snapshot
		var groupErr, singleErr error
		go func() { defer wg.Done(); groupSnaps, groupErr = s.SealGroup(ctx, []string{a.ID, b.ID}) }()
		go func() { defer wg.Done(); _, singleErr = other.SealBatch(ctx, b.ID) }()
		wg.Wait()

		if groupErr != nil || singleErr != nil {
			t.Fatalf("round %d: group=%v single=%v", i, groupErr, singleErr)
		}
		snapA, err := s.Snapshot(ctx, a.ID)
		if err != nil {
			t.Fatal(err)
		}
		snapB, err := s.Snapshot(ctx, b.ID)
		if err != nil {
			t.Fatal(err)
		}
		if snapA.Status != store.StatusSealed || snapB.Status != store.StatusSealed {
			t.Fatalf("round %d: expected both SEALED: %s / %s", i, snapA.Status, snapB.Status)
		}

		// The 200 group response has to agree with the database for both
		// members — the indivisible unit can never be reported half-open.
		stored := map[string]*store.Snapshot{a.ID: snapA, b.ID: snapB}
		for _, got := range groupSnaps {
			want := stored[got.ID]
			if got.Status != want.Status || got.SealedAt == nil ||
				!got.SealedAt.Equal(*want.SealedAt) {
				t.Fatalf("round %d: group response %+v disagrees with database %+v",
					i, got, want)
			}
		}

		// A second group seal must not move either sealedAt.
		again, err := s.SealGroup(ctx, []string{a.ID, b.ID})
		if err != nil {
			t.Fatal(err)
		}
		if !again[0].SealedAt.Equal(*snapA.SealedAt) || !again[1].SealedAt.Equal(*snapB.SealedAt) {
			t.Fatalf("round %d: sealedAt moved on reseal", i)
		}
	}
}

// waitForTupleLock polls until some backend is blocked on the batches row
// with batchID, returning false on timeout. PostgreSQL does not publish a
// lock holder's FOR UPDATE row lock in pg_locks — only the contender shows
// up, carrying a granted (prospective) tuple lock on that row together with
// an ungranted lock it is sleeping on (the holder's transactionid). So a
// waiter on a given row is: a backend whose granted tuple lock matches the
// row's ctid and which also holds any ungranted lock. The raw base pool
// defaults to the public search path while test tables live in an isolated
// schema, so names are qualified explicitly.
func waitForTupleLock(ctx context.Context, t *testing.T, schema, batchID string) bool {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		var waiting bool
		err := basePool.QueryRow(ctx, `
			SELECT EXISTS (
				SELECT 1
				FROM pg_locks tl
				JOIN `+schema+`.batches b
				  ON tl.page = (b.ctid::text::point)[0]::int
				 AND tl.tuple = (b.ctid::text::point)[1]::int
				WHERE tl.locktype = 'tuple' AND tl.granted = true
				  AND tl.relation = ($1 || '.batches')::regclass
				  AND b.id = $2
				  AND EXISTS (
					SELECT 1 FROM pg_locks w
					WHERE w.pid = tl.pid AND w.granted = false
				  )
			)`, schema, batchID).Scan(&waiting)
		if err != nil {
			t.Fatalf("scan lock state: %v", err)
		}
		if waiting {
			return true
		}
		time.Sleep(2 * time.Millisecond)
	}
	return false
}

// waitForQueuedTupleLock polls until a second backend is queued on the
// batches row's tuple lock behind the first waiter, returning false on
// timeout. The first waiter holds the prospective (granted) tuple lock and
// sleeps on the lock holder's transaction id; a backend arriving behind it
// carries an ungranted tuple lock on the same row.
func waitForQueuedTupleLock(ctx context.Context, t *testing.T, schema, batchID string) bool {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		var waiting bool
		err := basePool.QueryRow(ctx, `
			SELECT EXISTS (
				SELECT 1
				FROM pg_locks tl
				JOIN `+schema+`.batches b
				  ON tl.page = (b.ctid::text::point)[0]::int
				 AND tl.tuple = (b.ctid::text::point)[1]::int
				WHERE tl.locktype = 'tuple' AND tl.granted = false
				  AND tl.relation = ($1 || '.batches')::regclass
				  AND b.id = $2
			)`, schema, batchID).Scan(&waiting)
		if err != nil {
			t.Fatalf("scan lock state: %v", err)
		}
		if waiting {
			return true
		}
		time.Sleep(2 * time.Millisecond)
	}
	return false
}

// TestSealGroupReversedRequestsResponseMatchesDB deterministically reproduces
// the reversed-order group race: [A,B] and [B,A] validate while holding the
// same locks in series, then (in the buggy implementation) both applied their
// UPDATE outside the locking transaction. The loser's UPDATE matched zero
// rows yet it returned 200 with OPEN snapshots carrying no sealedAt while the
// database already showed SEALED — a self-contradictory handoff.
func TestSealGroupReversedRequestsResponseMatchesDB(t *testing.T) {
	ctx := context.Background()
	s, newPeer := newStore(ctx, t)
	peer := newPeer()
	defer peer.Close()

	a := mustBatch(ctx, t, s, 1)
	b := mustBatch(ctx, t, s, 1)
	for _, id := range []string{a.ID, b.ID} {
		if _, err := s.SubmitChunk(ctx, id, 1, []byte("x")); err != nil {
			t.Fatal(err)
		}
	}

	// Stage the interleaving with a blocker transaction on a raw connection:
	// hold B's row lock so G1 locks A and parks on B; G2 locks in the same
	// ascending order and queues right behind it.
	blocker, err := basePool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer blocker.Rollback(ctx)
	schema := testSchema(t)
	if _, err := blocker.Exec(ctx, "SET LOCAL search_path = "+schema); err != nil {
		t.Fatal(err)
	}
	if _, err := blocker.Exec(ctx, `SELECT id FROM batches WHERE id = $1 FOR UPDATE`, b.ID); err != nil {
		t.Fatal(err)
	}

	g1Done := make(chan struct{})
	var snapsAB []*store.Snapshot
	var errAB error
	go func() {
		defer close(g1Done)
		snapsAB, errAB = s.SealGroup(ctx, []string{a.ID, b.ID})
	}()
	if !waitForTupleLock(ctx, t, schema, b.ID) {
		t.Fatal("G1 never parked waiting for B's row lock")
	}

	g2Done := make(chan struct{})
	var snapsBA []*store.Snapshot
	var errBA error
	go func() {
		defer close(g2Done)
		snapsBA, errBA = peer.SealGroup(ctx, []string{b.ID, a.ID})
	}()

	// Release B: G1 wins, seals the group and returns; G2 then serialises
	// behind it and must observe the committed verdict.
	if err := blocker.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	select {
	case <-g1Done:
	case <-time.After(30 * time.Second):
		t.Fatal("winning group call deadlocked")
	}
	select {
	case <-g2Done:
	case <-time.After(30 * time.Second):
		t.Fatal("losing group call deadlocked")
	}

	if errAB != nil || errBA != nil {
		t.Fatalf("both reversed groups must succeed: %v / %v", errAB, errBA)
	}

	// Every 200 response snapshot must be SEALED, carry sealedAt, and agree
	// byte-for-byte with the database verdict.
	dbA, err := s.Snapshot(ctx, a.ID)
	if err != nil {
		t.Fatal(err)
	}
	dbB, err := s.Snapshot(ctx, b.ID)
	if err != nil {
		t.Fatal(err)
	}
	if dbA.Status != store.StatusSealed || dbB.Status != store.StatusSealed {
		t.Fatalf("database group not sealed: %s / %s", dbA.Status, dbB.Status)
	}
	assertResponseMatchesDB := func(label string, ordered []*store.Snapshot, want0, want1 string) {
		t.Helper()
		if len(ordered) != 2 || ordered[0].ID != want0 || ordered[1].ID != want1 {
			t.Fatalf("%s: snapshots not in request order: %+v", label, ordered)
		}
		for _, snap := range ordered {
			if snap.Status != store.StatusSealed || snap.SealedAt == nil {
				t.Fatalf("%s: 200 response reports member %s still OPEN (status=%s sealedAt=%v) "+
					"but the database shows it sealed", label, snap.ID, snap.Status, snap.SealedAt)
			}
		}
		if !ordered[0].SealedAt.Equal(*dbA.SealedAt) || !ordered[1].SealedAt.Equal(*dbB.SealedAt) {
			t.Fatalf("%s: response sealedAt disagrees with database", label)
		}
	}
	assertResponseMatchesDB("[A,B]", snapsAB, a.ID, b.ID)
	assertResponseMatchesDB("[B,A]", snapsBA, b.ID, a.ID)
}

// TestSealGroupVsSingleSealDeterministic stages the documented interleave:
// the group has locked every member and passed its post-lock validation
// (still inside the locking transaction) while B's single-batch seal waits
// on the group's row lock. The group UPDATE therefore has to be atomic with
// the validation transaction; if it is split off after the locks release,
// the queued SealBatch slips in first and seals B on its own, after which the
// pooled group UPDATE only seals A — the two members end up sealed by
// different operations and the group can be observed half sealed.
func TestSealGroupVsSingleSealDeterministic(t *testing.T) {
	ctx := context.Background()
	s, newPeer := newStore(ctx, t)
	peer := newPeer()
	defer peer.Close()

	first := mustBatch(ctx, t, s, 1)
	second := mustBatch(ctx, t, s, 1)
	for _, id := range []string{first.ID, second.ID} {
		if _, err := s.SubmitChunk(ctx, id, 1, []byte("x")); err != nil {
			t.Fatal(err)
		}
	}
	// SealGroup locks rows in ascending ID order. To make the group park
	// while still holding the single seal's row, the blocker must hold the
	// HIGH id (last locked) and the single seal targets the LOW id (already
	// locked by the parked group).
	loID, hiID := first.ID, second.ID
	if loID > hiID {
		loID, hiID = hiID, loID
	}

	blocker, err := basePool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer blocker.Rollback(ctx)
	schema := testSchema(t)
	if _, err := blocker.Exec(ctx, "SET LOCAL search_path = "+schema); err != nil {
		t.Fatal(err)
	}
	if _, err := blocker.Exec(ctx, `SELECT id FROM batches WHERE id = $1 FOR UPDATE`, hiID); err != nil {
		t.Fatal(err)
	}

	groupDone := make(chan struct{})
	var groupSnaps []*store.Snapshot
	var groupErr error
	go func() {
		defer close(groupDone)
		groupSnaps, groupErr = s.SealGroup(ctx, []string{first.ID, second.ID})
	}()
	if !waitForTupleLock(ctx, t, schema, hiID) {
		t.Fatal("group never parked waiting for the high row lock")
	}

	singleDone := make(chan struct{})
	var singleErr error
	go func() {
		defer close(singleDone)
		_, singleErr = peer.SealBatch(ctx, loID)
	}()

	// The single seal must be queued behind the group on the low row.
	if !waitForTupleLock(ctx, t, schema, loID) {
		t.Fatal("single-batch seal never queued behind the group")
	}

	if err := blocker.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	select {
	case <-groupDone:
	case <-time.After(30 * time.Second):
		t.Fatal("group seal deadlocked")
	}
	select {
	case <-singleDone:
	case <-time.After(30 * time.Second):
		t.Fatal("single seal deadlocked")
	}

	if groupErr != nil || singleErr != nil {
		t.Fatalf("group=%v single=%v", groupErr, singleErr)
	}

	dbLo, err := s.Snapshot(ctx, loID)
	if err != nil {
		t.Fatal(err)
	}
	dbHi, err := s.Snapshot(ctx, hiID)
	if err != nil {
		t.Fatal(err)
	}
	if dbLo.Status != store.StatusSealed || dbHi.Status != store.StatusSealed {
		t.Fatalf("expected both sealed: %s / %s", dbLo.Status, dbHi.Status)
	}
	// The group sealed both members in one statement while holding the
	// locks, so both timestamps come from the same group operation.
	if !dbLo.SealedAt.Equal(*dbHi.SealedAt) {
		t.Fatalf("members sealed by different operations: %v vs %v",
			dbLo.SealedAt, dbHi.SealedAt)
	}

	// The group's 200 snapshots must match the database verdict for every
	// member (request order is the created order [first, second]).
	if len(groupSnaps) != 2 || groupSnaps[0].ID != first.ID || groupSnaps[1].ID != second.ID {
		t.Fatalf("group snapshots not in request order: %+v", groupSnaps)
	}
	dbByID := map[string]*store.Snapshot{loID: dbLo, hiID: dbHi}
	for _, snap := range groupSnaps {
		db := dbByID[snap.ID]
		if snap.Status != store.StatusSealed || snap.SealedAt == nil {
			t.Fatalf("group response member %s reports %s without sealedAt", snap.ID, snap.Status)
		}
		if !snap.SealedAt.Equal(*db.SealedAt) {
			t.Fatalf("member %s response sealedAt %v disagrees with database %v",
				snap.ID, snap.SealedAt, db.SealedAt)
		}
	}
}

// TestSealGroupSeesChunkCommittedDuringLockWait stages the interleave where
// B's final chunk wins the row lock and commits while the group request is
// still parked on that lock: the chunk's critical section (lock B's row,
// insert the chunk) is replayed on a raw connection and held open, the group
// seal queues behind it, and only then does the chunk commit. The group's
// verdict must be computed from the state after the locks are acquired, so
// it observes the committed chunk and seals the group. A verdict snapshot
// pinned before the lock wait (REPEATABLE READ) would instead report the
// just-committed seq as a gap and fail with ErrIncomplete, leaving B
// complete but OPEN — the caller then misreads the group as unsealable.
func TestSealGroupSeesChunkCommittedDuringLockWait(t *testing.T) {
	ctx := context.Background()
	s, _ := newStore(ctx, t)

	a := mustBatch(ctx, t, s, 1)
	b := mustBatch(ctx, t, s, 2)
	if _, err := s.SubmitChunk(ctx, a.ID, 1, []byte("a")); err != nil {
		t.Fatal(err)
	}
	if _, err := s.SubmitChunk(ctx, b.ID, 1, []byte("b1")); err != nil {
		t.Fatal(err)
	}
	// B is missing only its final chunk (seq 2).

	// Replay SubmitChunk's critical section for B's final chunk on a raw
	// connection, holding the transaction open between the insert and the
	// commit: the chunk occupies B's row lock first, exactly like a
	// concurrent submission that won the lock race.
	schema := testSchema(t)
	chunkTx, err := basePool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer chunkTx.Rollback(ctx)
	if _, err := chunkTx.Exec(ctx, "SET LOCAL search_path = "+schema); err != nil {
		t.Fatal(err)
	}
	if _, err := chunkTx.Exec(ctx,
		`SELECT expected_chunks, status FROM batches WHERE id = $1 FOR UPDATE`, b.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := chunkTx.Exec(ctx,
		`INSERT INTO chunks (batch_id, seq, payload) VALUES ($1, $2, $3)`,
		b.ID, 2, []byte("b2")); err != nil {
		t.Fatal(err)
	}

	// The group seal locks A, then parks on B behind the uncommitted chunk.
	// Being parked proves its first statement already started, so a
	// transaction-scoped snapshot would already be frozen without the chunk.
	groupDone := make(chan struct{})
	var snaps []*store.Snapshot
	var groupErr error
	go func() {
		defer close(groupDone)
		snaps, groupErr = s.SealGroup(ctx, []string{a.ID, b.ID})
	}()
	if !waitForTupleLock(ctx, t, schema, b.ID) {
		t.Fatal("group seal never parked waiting for B's row lock")
	}

	// The final chunk commits during the group's lock wait; the group then
	// acquires its locks and must see the complete chunk set.
	if err := chunkTx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	select {
	case <-groupDone:
	case <-time.After(30 * time.Second):
		t.Fatal("group seal deadlocked")
	}

	if groupErr != nil {
		t.Fatalf("group waited for the final chunk commit, so it must seal; got %v", groupErr)
	}
	if len(snaps) != 2 || snaps[0].ID != a.ID || snaps[1].ID != b.ID {
		t.Fatalf("snapshots not in request order: %+v", snaps)
	}
	for _, snap := range snaps {
		if snap.Status != store.StatusSealed || snap.SealedAt == nil || len(snap.Gaps) != 0 {
			t.Fatalf("member not cleanly sealed: %+v", snap)
		}
	}
	if !snaps[0].SealedAt.Equal(*snaps[1].SealedAt) {
		t.Fatalf("members not sealed by one transaction: %v vs %v",
			snaps[0].SealedAt, snaps[1].SealedAt)
	}

	// The committed database state agrees with the 200 response.
	for i, id := range []string{a.ID, b.ID} {
		db, err := s.Snapshot(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		if db.Status != store.StatusSealed || len(db.Gaps) != 0 ||
			db.SealedAt == nil || !db.SealedAt.Equal(*snaps[i].SealedAt) {
			t.Fatalf("database disagrees with group response: db=%+v resp=%+v", db, snaps[i])
		}
	}
}

// TestSealGroupIncompleteReportsVerdictSnapshot stages the handoff race in
// which the group seal adjudicates B incomplete while B's final chunk is
// queued behind the group's row locks: the chunk commits the moment the
// group transaction ends. The INCOMPLETE result must describe the state as
// adjudicated under the locks — B OPEN with gaps [2] — not the post-release
// state in which B is already complete (or, with a follow-up seal in
// between, even SEALED). A failure response that names no incomplete member
// contradicts every observable state and hides which chunk was missing, so
// the caller cannot tell the seal condition apart from a lost chunk.
func TestSealGroupIncompleteReportsVerdictSnapshot(t *testing.T) {
	ctx := context.Background()
	s, newPeer := newStore(ctx, t)
	peer := newPeer()
	defer peer.Close()

	// Padding members (all complete) precede B in request order. An
	// implementation that re-reads the members after releasing the locks
	// walks them in request order, so a long prefix guarantees the queued
	// chunk has landed before B is re-observed — turning a stale
	// post-release read from a coin flip into a certainty.
	pads := make([]*store.Batch, 0, 8)
	for range 8 {
		p := mustBatch(ctx, t, s, 1)
		if _, err := s.SubmitChunk(ctx, p.ID, 1, []byte("p")); err != nil {
			t.Fatal(err)
		}
		pads = append(pads, p)
	}
	a := mustBatch(ctx, t, s, 1)
	b := mustBatch(ctx, t, s, 2)
	if _, err := s.SubmitChunk(ctx, a.ID, 1, []byte("a")); err != nil {
		t.Fatal(err)
	}
	if _, err := s.SubmitChunk(ctx, b.ID, 1, []byte("b1")); err != nil {
		t.Fatal(err)
	}
	// B is missing only its final chunk (seq 2).

	// Hold B's row lock on a raw connection: the group seal parks on it,
	// and the final chunk submitted afterwards queues behind the group.
	schema := testSchema(t)
	blocker, err := basePool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer blocker.Rollback(ctx)
	if _, err := blocker.Exec(ctx, "SET LOCAL search_path = "+schema); err != nil {
		t.Fatal(err)
	}
	if _, err := blocker.Exec(ctx, `SELECT id FROM batches WHERE id = $1 FOR UPDATE`, b.ID); err != nil {
		t.Fatal(err)
	}

	ids := make([]string, 0, len(pads)+2)
	for _, p := range pads {
		ids = append(ids, p.ID)
	}
	ids = append(ids, a.ID, b.ID)

	groupDone := make(chan struct{})
	var snaps []*store.Snapshot
	var groupErr error
	go func() {
		defer close(groupDone)
		snaps, groupErr = s.SealGroup(ctx, ids)
	}()
	if !waitForTupleLock(ctx, t, schema, b.ID) {
		t.Fatal("group seal never parked waiting for B's row lock")
	}

	chunkDone := make(chan struct{})
	var chunkErr error
	go func() {
		defer close(chunkDone)
		_, chunkErr = peer.SubmitChunk(ctx, b.ID, 2, []byte("b2"))
	}()
	if !waitForQueuedTupleLock(ctx, t, schema, b.ID) {
		t.Fatal("final chunk never queued behind the group seal")
	}

	// Release the blocker: the group acquires its locks while the chunk is
	// still queued behind it, so the verdict must be INCOMPLETE with B
	// missing seq 2; the chunk commits right after the group transaction.
	if err := blocker.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	select {
	case <-groupDone:
	case <-time.After(30 * time.Second):
		t.Fatal("group seal deadlocked")
	}
	select {
	case <-chunkDone:
	case <-time.After(30 * time.Second):
		t.Fatal("queued final chunk deadlocked")
	}
	if chunkErr != nil {
		t.Fatalf("final chunk: %v", chunkErr)
	}

	if !errors.Is(groupErr, store.ErrIncomplete) {
		t.Fatalf("group adjudicated while the final chunk was still queued, "+
			"so it must report INCOMPLETE; got %v", groupErr)
	}
	if len(snaps) != len(ids) {
		t.Fatalf("want %d member snapshots in request order, got %d", len(ids), len(snaps))
	}
	for i, snap := range snaps {
		if snap.ID != ids[i] {
			t.Fatalf("snapshots not in request order at %d: %+v", i, snaps)
		}
	}
	// The failure must name the member and the chunk missing at verdict
	// time — B OPEN, received 1, gaps [2] — not the post-release state in
	// which B is already complete.
	last := snaps[len(snaps)-1]
	if last.Status != store.StatusOpen || last.Received != 1 ||
		len(last.Gaps) != 1 || last.Gaps[0] != 2 {
		t.Fatalf("INCOMPLETE must report B as adjudicated (OPEN, gaps [2]); got %+v", last)
	}
	for _, snap := range snaps[:len(snaps)-1] {
		if len(snap.Gaps) != 0 {
			t.Fatalf("complete member reported gaps: %+v", snap)
		}
	}

	// The queued chunk landed right after the verdict: B is complete but
	// still OPEN — exactly what the caller observes on the next query, and
	// consistent with the 409 it just received.
	db, err := s.Snapshot(ctx, b.ID)
	if err != nil {
		t.Fatal(err)
	}
	if db.Status != store.StatusOpen || len(db.Gaps) != 0 {
		t.Fatalf("post-handoff state should be complete and OPEN: %+v", db)
	}

	// A follow-up group seal now succeeds; the earlier 409 stays truthful
	// because it described the state at its own verdict point.
	sealed, err := s.SealGroup(ctx, ids)
	if err != nil {
		t.Fatalf("follow-up group seal: %v", err)
	}
	for _, snap := range sealed {
		if snap.Status != store.StatusSealed {
			t.Fatalf("follow-up seal left member OPEN: %+v", snap)
		}
	}
}

// TestRestartConsistency closes every connection and opens a fresh store,
// modelling a full restart of all API instances: acknowledgements, gaps and
// sealing verdict must survive unchanged.
func TestRestartConsistency(t *testing.T) {
	ctx := context.Background()
	s, newPeer := newStore(ctx, t)

	open := mustBatch(ctx, t, s, 3)
	sealed := mustBatch(ctx, t, s, 2)

	if _, err := s.SubmitChunk(ctx, open.ID, 1, []byte("o1")); err != nil {
		t.Fatal(err)
	}
	if _, err := s.SubmitChunk(ctx, open.ID, 3, []byte("o3")); err != nil {
		t.Fatal(err)
	}
	openAck, err := s.SubmitChunk(ctx, open.ID, 1, []byte("o1"))
	if err != nil {
		t.Fatal(err)
	}

	for seq, p := range map[int]string{1: "s1", 2: "s2"} {
		if _, err := s.SubmitChunk(ctx, sealed.ID, seq, []byte(p)); err != nil {
			t.Fatal(err)
		}
	}
	sealedSnap, err := s.SealBatch(ctx, sealed.ID)
	if err != nil {
		t.Fatal(err)
	}

	// Simulate full API restart: drop all pools and reconnect.
	s.Close()
	restarted := newPeer()
	defer restarted.Close()

	gotOpen, err := restarted.Snapshot(ctx, open.ID)
	if err != nil {
		t.Fatal(err)
	}
	if gotOpen.Status != store.StatusOpen || gotOpen.Received != 2 {
		t.Fatalf("open batch state not preserved after restart: %+v", gotOpen)
	}
	if len(gotOpen.Gaps) != 1 || gotOpen.Gaps[0] != 2 {
		t.Fatalf("gaps not preserved after restart: %+v", gotOpen)
	}

	// The retransmission acknowledgement after restart must still be the
	// original confirmation, not a new record.
	ack2, err := restarted.SubmitChunk(ctx, open.ID, 1, []byte("o1"))
	if err != nil {
		t.Fatal(err)
	}
	if ack2.Created || !ack2.ReceivedAt.Equal(openAck.ReceivedAt) {
		t.Fatalf("acknowledgement changed after restart: %+v vs %+v", ack2, openAck)
	}

	gotSealed, err := restarted.Snapshot(ctx, sealed.ID)
	if err != nil {
		t.Fatal(err)
	}
	if gotSealed.Status != store.StatusSealed || !gotSealed.SealedAt.Equal(*sealedSnap.SealedAt) {
		t.Fatalf("sealed verdict changed after restart: %+v", gotSealed)
	}
	if _, err := restarted.SealBatch(ctx, sealed.ID); err != nil {
		t.Fatalf("repeated seal after restart should be idempotent: %v", err)
	}
}
