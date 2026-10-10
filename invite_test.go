package main

import (
	"context"
	"errors"
	"testing"

	"maunium.net/go/mautrix/id"

	"gorkbot/llm/testutil"
)

const (
	testGork     = id.UserID("@gork:example.org")
	testInviter  = id.UserID("@friend:example.org")
	testStranger = id.UserID("@stranger:elsewhere.org")
)

// newInviteBot builds a Bot whose room join is recorded instead of sent to a
// homeserver. The whitelist is the list of inviters gork will auto-join for.
func newInviteBot(t *testing.T, joinErr error, whitelist ...id.UserID) (*Bot, *[]id.RoomID) {
	t.Helper()
	b := newTestBot(t, &testutil.Client{}, nil, nil)
	b.selfID = testGork
	b.allowedInviters = make(map[id.UserID]struct{}, len(whitelist))
	for _, user := range whitelist {
		b.allowedInviters[user] = struct{}{}
	}
	joined := &[]id.RoomID{}
	b.joinRoomFn = func(_ context.Context, room id.RoomID) error {
		*joined = append(*joined, room)
		return joinErr
	}
	return b, joined
}

func TestHandleInviteJoinsWhitelistedInviter(t *testing.T) {
	b, joined := newInviteBot(t, nil, testInviter)

	b.HandleInvite(context.Background(), inviteEvent(t, testInviter, testGork, 1))

	if len(*joined) != 1 || (*joined)[0] != testRoom {
		t.Fatalf("expected join of %s, got %v", testRoom, *joined)
	}
}

func TestHandleInviteIgnoresUnlistedInviter(t *testing.T) {
	b, joined := newInviteBot(t, nil, testInviter)

	b.HandleInvite(context.Background(), inviteEvent(t, testStranger, testGork, 1))

	if len(*joined) != 0 {
		t.Fatalf("expected no join for unlisted inviter, got %v", *joined)
	}
}

func TestHandleInviteIgnoresEmptyWhitelist(t *testing.T) {
	b, joined := newInviteBot(t, nil)

	b.HandleInvite(context.Background(), inviteEvent(t, testInviter, testGork, 1))

	if len(*joined) != 0 {
		t.Fatalf("expected no join with an empty whitelist, got %v", *joined)
	}
}

func TestHandleInviteIgnoresOtherEvents(t *testing.T) {
	b, joined := newInviteBot(t, nil, testInviter)

	// A join from the whitelisted user is not an invite.
	b.HandleInvite(context.Background(), memberEvent(t, testInviter, 1))
	// An invite addressed to somebody else is not for us.
	b.HandleInvite(context.Background(), inviteEvent(t, testInviter, "@other:example.org", 2))
	// A plain message from the whitelisted user is not a member event.
	b.HandleInvite(context.Background(), textEvent(t, testInviter, "join please", 3))

	if len(*joined) != 0 {
		t.Fatalf("expected no join for non-invite events, got %v", *joined)
	}
}

func TestHandleInviteJoinFailureDoesNotPanic(t *testing.T) {
	b, joined := newInviteBot(t, errors.New("boom"), testInviter)

	b.HandleInvite(context.Background(), inviteEvent(t, testInviter, testGork, 1))

	if len(*joined) != 1 {
		t.Fatalf("expected one join attempt, got %v", *joined)
	}
}

func TestInviterAllowed(t *testing.T) {
	b, _ := newInviteBot(t, nil, testInviter)

	if !b.inviterAllowed(testInviter) {
		t.Errorf("expected %s to be allowed", testInviter)
	}
	if b.inviterAllowed(testStranger) {
		t.Errorf("expected %s not to be allowed", testStranger)
	}
}
