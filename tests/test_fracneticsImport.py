import pytest

def test_import_fracnetics():
    """
    Test import of 'fracnetics'
    """
    try:
        import fracnetics
    except Exception as e:
        pytest.fail(f"❌ Error importing'fracnetics': {e}")

if __name__ == "__main__":
    test_import_fracnetics()

