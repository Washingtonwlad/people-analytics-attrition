"""Smoke tests for every public Streamlit view."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "app.py"
SECTIONS = [
    "Evidence overview",
    "Behavioral evidence",
    "Model validation",
    "Responsible-use audit",
    "Methods and limitations",
]


def assert_app_has_no_exceptions(app_test: AppTest, section: str) -> None:
    """Expose Streamlit exceptions with the section that produced them."""
    assert not app_test.exception, (
        f"Streamlit raised an exception in {section}: "
        f"{[exception.value for exception in app_test.exception]}"
    )


def test_every_streamlit_section_loads():
    """Run the application and navigate through all five aggregate views."""
    app_test = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert_app_has_no_exceptions(app_test, SECTIONS[0])

    section_control = app_test.sidebar.radio[0]
    for section in SECTIONS[1:]:
        section_control.set_value(section)
        app_test.run()
        assert_app_has_no_exceptions(app_test, section)
