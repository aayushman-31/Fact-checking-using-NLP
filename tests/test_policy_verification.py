import tempfile
import unittest
from pathlib import Path

import main
from verification.verifier import verify_claim


class PolicyClaimVerificationTests(unittest.TestCase):
    def test_supports_coverage_claim(self):
        evidence = ["The policy covers emergency room visits and hospital stays."]
        self.assertEqual(
            verify_claim("The policy covers emergency room visits.", evidence),
            "SUPPORTED",
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


if __name__ == "__main__":
    unittest.main()
