from streamlit.testing.v1 import AppTest
from preprocess import preprocess

def run_app(text):
    at = AppTest.from_file("app.py", default_timeout=60).run()
    at.text_area[0].input(text)
    at.button[0].click().run()
    assert not at.exception
    return at

def test_preprocess():
    assert preprocess("<b>The TEAMS</b> won 2 matches!") == "team match"

def test_empty_input():
    assert len(run_app("").warning) == 1

def test_sports_news():
    at = run_app("The striker scored two goals as his team won the football championship final")
    assert "Sports" in at.success[0].value