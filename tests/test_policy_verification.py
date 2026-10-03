from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path

import fitz

import main
from document.document_loader import load_document
from verification.verifier import verify_claim
from ner.entity_extractor import _extract_entities_with_rules


class PolicyClaimVerificationTests(unittest.TestCase):
    def test_loads_text_from_pdf_document(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            document_path = Path(tmpdir) / "policy.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            page.insert_text((72, 72), "The policy covers emergency room visits.")
            pdf.save(document_path)
            pdf.close()

            self.assertIn("The policy covers emergency room visits.", load_document(document_path))

    def test_rule_based_entity_fallback_extracts_proper_names(self):
        entities = _extract_entities_with_rules("John Smith needs coverage in California.")
        self.assertEqual(
            entities,
            [
                {"text": "John Smith", "label": "PROPER_NOUN"},
                {"text": "California", "label": "PROPER_NOUN"},
            ],
        )

    def test_supports_coverage_claim(self):
        evidence = ["The policy covers emergency room visits and hospital stays."]
        self.assertEqual(
            verify_claim("The policy covers emergency room visits.", evidence),
            "SUPPORTED",
        )

    def test_unrelated_coverage_does_not_support_claim(self):
        evidence = ["The policy covers emergency room visits and hospital stays."]
        self.assertEqual(
            verify_claim("The policy covers dental care.", evidence),
            "NEI",
        )

    def test_refutes_exclusion_claim(self):
        evidence = ["The policy excludes pre-existing conditions during the first 12 months."]
        self.assertEqual(
            verify_claim("The policy covers pre-existing conditions immediately.", evidence),
            "REFUTED",
        )

    def test_handles_unspecified_policy_claims(self):
        evidence = ["The policy summary does not mention dental care benefits."]
        self.assertEqual(
            verify_claim("The policy includes dental coverage.", evidence),
            "NEI",
        )

    def test_matches_generic_coverage_topic(self):
        evidence = ["The plan covers dental care and routine cleanings."]
        self.assertEqual(
            verify_claim("The policy includes dental benefits.", evidence),
            "SUPPORTED",
        )

    def test_matches_generic_exclusion_topic(self):
        evidence = ["The policy excludes cosmetic surgery and elective procedures."]
        self.assertEqual(
            verify_claim("The policy covers cosmetic surgery.", evidence),
            "REFUTED",
        )

    def test_retrieves_evidence_from_document_sentences(self):
        document_sentences = [
            "The policy covers emergency room visits and hospital stays.",
            "This plan excludes pre-existing conditions during the first year.",
            "Routine dental cleanings are not listed in the summary."
        ]

        evidence = __import__("retrieval.hybrid_retriever", fromlist=["retrieve_evidence"]).retrieve_evidence(
            "The policy covers emergency room visits.",
            document_sentences=document_sentences,
            top_k=2,
        )

        self.assertIn("The policy covers emergency room visits and hospital stays.", evidence)

    def test_evaluate_policy_match_returns_matching_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            document_path = Path(tmpdir) / "policy.txt"
            document_path.write_text(
                "The policy covers emergency room visits and hospital stays.\n"
                "The plan excludes cosmetic surgery procedures.\n",
                encoding="utf-8",
            )
            self.assertEqual(
                main.evaluate_policy_match("emergency room visits", document_path),
                "This document matches your needs",
            )

    def test_evaluate_policy_match_returns_not_cover_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            document_path = Path(tmpdir) / "policy.txt"
            document_path.write_text(
                "The policy excludes cosmetic surgery procedures.\n"
                "The plan covers emergency room visits.\n",
                encoding="utf-8",
            )
            self.assertEqual(
                main.evaluate_policy_match("cosmetic surgery", document_path),
                "This policy does not cover that.",
            )

    def test_question_for_unrelated_coverage_does_not_match_generic_policy_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            document_path = Path(tmpdir) / "policy.txt"
            document_path.write_text(
                "Universal Sompo General Insurance provides coverage for hospitalization.",
                encoding="utf-8",
            )
            self.assertEqual(
                main.evaluate_policy_match(
                    "does the insurance cover divorce lawyer's fee",
                    document_path,
                ),
                "This policy does not cover that.",
            )

    def test_question_form_matches_hospitalization_payment_clause(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            document_path = Path(tmpdir) / "policy.txt"
            document_path.write_text(
                "We will pay reasonable expenses for emergency hospitalization.",
                encoding="utf-8",
            )
            self.assertEqual(
                main.evaluate_policy_match(
                    "does the insurance cover emergency hospital visits",
                    document_path,
                ),
                "This document matches your needs",
            )

    def test_one_shared_generic_word_does_not_match_multiword_need(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            document_path = Path(tmpdir) / "policy.txt"
            document_path.write_text(
                "This insurance provides coverage for eligible medical expenses.",
                encoding="utf-8",
            )
            self.assertEqual(
                main.evaluate_policy_match("does it cover marriage expenses", document_path),
                "This policy does not cover that.",
            )

    def test_negative_coverage_question_reports_that_policy_covers_need(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            document_path = Path(tmpdir) / "policy.txt"
            document_path.write_text(
                "The policy covers emergency room visits.",
                encoding="utf-8",
            )
            self.assertEqual(
                main.evaluate_policy_match(
                    "does the insurance not cover emergency hospital visits",
                    document_path,
                ),
                "This policy covers that.",
            )

    def test_cli_prints_the_policy_sentence_used_as_evidence(self):
        evidence_sentence = "The policy covers emergency room visits and hospital stays."
        with tempfile.TemporaryDirectory() as tmpdir:
            document_path = Path(tmpdir) / "policy.txt"
            document_path.write_text(evidence_sentence, encoding="utf-8")
            output = StringIO()

            with redirect_stdout(output):
                main.main([
                    "--coverage-need", "emergency room visits",
                    "--policy-document", str(document_path),
                ])

        self.assertIn("This document matches your needs", output.getvalue())
        self.assertIn(evidence_sentence, output.getvalue())


if __name__ == "__main__":
    unittest.main()
