from ..logging import PRINT, _IndentPrinter, logging_indent


class TestLoggingIndent:

    def test_recover_builtin_print(self):
        assert print == PRINT  # noqa
        assert _IndentPrinter.level == 0
        with logging_indent():
            # partial
            assert print.func == _IndentPrinter.print_body  # noqa
            assert _IndentPrinter.level == 1
            with logging_indent():
                assert print.func == _IndentPrinter.print_body  # noqa
                assert _IndentPrinter.level == 2
            assert print.func == _IndentPrinter.print_body  # noqa
            assert _IndentPrinter.level == 1

        assert print == PRINT  # noqa
        assert _IndentPrinter.level == 0
