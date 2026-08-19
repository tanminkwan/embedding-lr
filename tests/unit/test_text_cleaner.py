from embedding_lr.preprocessing.text_cleaner import clean_text


class TestCleanText:
    def test_strips_fence_delimiters_with_language_tag_and_keeps_body(self):
        text = "확인 순서:\n```bash\nnginx -t\n```\n끝"

        result = clean_text(text)

        assert "```" not in result
        assert "nginx -t" in result

    def test_strips_fence_delimiters_without_language_tag_and_keeps_body(self):
        text = "```\nshow log traffic\n```"

        result = clean_text(text)

        assert "```" not in result
        assert "show log traffic" in result

    def test_removes_exception_stack_trace_line(self):
        text = "에러 발생\njava.lang.NullPointerException: foo\n다음 단계"

        result = clean_text(text)

        assert "NullPointerException" not in result
        assert "에러 발생" in result
        assert "다음 단계" in result

    def test_removes_caused_by_line(self):
        text = "본문\nCaused by: java.io.IOException\n나머지"

        result = clean_text(text)

        assert "Caused by" not in result

    def test_removes_java_stack_frame_line(self):
        text = "본문\n\tat com.example.Foo.bar(Foo.java:42)\n나머지"

        result = clean_text(text)

        assert "at com.example.Foo.bar" not in result

    def test_removes_traceback_line(self):
        text = "본문\nTraceback (most recent call last):\n나머지"

        result = clean_text(text)

        assert "Traceback" not in result

    def test_does_not_false_positive_on_plain_sentence_containing_exception_word(self):
        text = "Exception 처리 방법 알려줘"

        result = clean_text(text)

        assert result == text

    def test_normalizes_repeated_spaces(self):
        text = "확인    순서를     알려줘"

        result = clean_text(text)

        assert result == "확인 순서를 알려줘"

    def test_normalizes_repeated_newlines(self):
        text = "본문1\n\n\n\n본문2"

        result = clean_text(text)

        assert result == "본문1\n본문2"

    def test_strips_leading_and_trailing_whitespace(self):
        text = "  \n 본문 \n  "

        result = clean_text(text)

        assert result == "본문"

    def test_applies_all_rules_together_in_fixed_order(self):
        text = (
            "확인 순서:\n"
            "```bash\n"
            "실행\n"
            "```\n"
            "\n"
            "java.lang.RuntimeException: boom\n"
            "\n"
            "   추가   공백   정리"
        )

        result = clean_text(text)

        assert "```" not in result
        assert "RuntimeException" not in result
        assert "실행" in result
        assert "  " not in result
        assert "\n\n" not in result

    def test_empty_string_returns_empty_string(self):
        assert clean_text("") == ""

    def test_noop_on_already_clean_text(self):
        text = "이미 깨끗한 텍스트입니다"

        assert clean_text(text) == text
