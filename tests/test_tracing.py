from triage import tracing


def test_mask_redacts_emails_and_tokens_in_nested_data():
    data = {"body": "mail me at jane.doe@example.com", "msgs": ["key ghp_abcdefghijklmnop1234 here"]}
    out = tracing.mask(data=data)
    assert out == {"body": "mail me at <email>", "msgs": ["key <token> here"]}


def test_untraced_run_is_a_no_op(monkeypatch):
    monkeypatch.setattr(tracing, "client", lambda: None)
    with tracing.traced_run(1) as run:
        run.score("x", 1.0)
        run.finish("done")
    assert run.callbacks == [] and run.trace_id is None
