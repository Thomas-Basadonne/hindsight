"""Reply context: a short user reply carries an excerpt of the agent reply it answers,
so the extractor can resolve «la 1», «1) sì 2) no» even when the referent was shipped in an
earlier append batch. Off by default: upstream payload unchanged."""

import json

import hindsight_hermes as plugin  # noqa: F401  (loaded by conftest)


def _messages(fake, index=0):
    return [m["content"] for m in json.loads(fake.retains[index]["items"][0]["content"])]


def test_off_by_default_payload_is_unchanged(provider):
    instance, fake = provider({"retain_every_n_turns": 2})
    instance.sync_turn("proponi", "1) A 2) B")
    instance.sync_turn("la 1", "ok, A")
    instance.shutdown()
    assert _messages(fake) == ["User: proponi", "Assistant: 1) A 2) B", "User: la 1", "Assistant: ok, A"]


def test_short_reply_quotes_the_previous_agent_reply_across_batches(provider):
    instance, fake = provider({"retain_every_n_turns": 1, "retain_reply_context_chars": 1000})
    instance.sync_turn("proponi", "1) A 2) B")
    instance.sync_turn("la 1", "ok, A")
    instance.shutdown()
    assert _messages(fake, 0) == ["User: proponi", "Assistant: 1) A 2) B"]  # first turn: nothing to quote
    assert _messages(fake, 1) == ["User (replying to Assistant: «1) A 2) B»): la 1", "Assistant: ok, A"]


def test_the_agent_reply_is_stored_once_only_quoted(provider):
    """No duplication: the agent reply appears as an Assistant message exactly once overall."""
    instance, fake = provider({"retain_every_n_turns": 1, "retain_reply_context_chars": 1000})
    instance.sync_turn("proponi", "1) A 2) B")
    instance.sync_turn("la 1", "ok, A")
    instance.shutdown()
    all_msgs = [m for i in range(len(fake.retains)) for m in _messages(fake, i)]
    assert sum(m == "Assistant: 1) A 2) B" for m in all_msgs) == 1


def test_only_the_first_turn_of_a_batch_is_annotated(provider):
    """Inside a batch the replied-to message is adjacent: no quote (no second copy to restate)."""
    instance, fake = provider({"retain_every_n_turns": 2, "retain_reply_context_chars": 1000})
    instance.sync_turn("proponi", "1) A 2) B")
    instance.sync_turn("la 1", "ok, A")  # same batch as its referent
    instance.sync_turn("sì", "fatto")  # first turn of batch 2: referent «ok, A» already shipped
    instance.sync_turn("grazie", "prego")
    instance.shutdown()
    assert _messages(fake, 0)[2] == "User: la 1"
    assert _messages(fake, 1)[0] == "User (replying to Assistant: «ok, A»): sì"
    assert _messages(fake, 1)[2] == "User: grazie"


def test_long_replies_are_not_annotated(provider):
    instance, fake = provider(
        {"retain_every_n_turns": 2, "retain_reply_context_chars": 1000, "retain_reply_short_chars": 20}
    )
    instance.sync_turn("proponi", "1) A 2) B")
    instance.sync_turn("x" * 21, "ok")
    instance.shutdown()
    assert _messages(fake)[2] == "User: " + "x" * 21


def test_long_previous_reply_is_cut_to_head_and_tail(provider):
    prev = "H" * 600 + "M" * 2000 + "T" * 900
    instance, fake = provider({"retain_every_n_turns": 1, "retain_reply_context_chars": 1000})
    instance.sync_turn("proponi", prev)
    instance.sync_turn("la 1", "ok")
    instance.shutdown()
    quoted = _messages(fake, 1)[0]
    assert quoted.startswith("User (replying to Assistant: «" + "H" * 400 + " … " + "T" * 600 + "»): la 1")
    assert "M" not in quoted


def test_session_switch_never_quotes_the_old_session(provider):
    instance, fake = provider({"retain_every_n_turns": 1, "retain_reply_context_chars": 1000})
    instance.sync_turn("proponi", "1) A 2) B")
    instance.on_session_switch("session-2", reset=True)
    instance.sync_turn("la 1", "?", session_id="session-2")
    instance.shutdown()
    assert _messages(fake, len(fake.retains) - 1)[0] == "User: la 1"


def test_the_label_is_configurable(provider):
    instance, fake = provider(
        {"retain_every_n_turns": 1, "retain_reply_context_chars": 1000, "retain_reply_context_label": "in risposta a"}
    )
    instance.sync_turn("proponi", "1) A 2) B")
    instance.sync_turn("la 1", "ok, A")
    instance.shutdown()
    assert _messages(fake, 1)[0] == "User (in risposta a Assistant: «1) A 2) B»): la 1"
