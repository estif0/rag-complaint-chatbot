"""
Unit tests for the RAGEvaluator module.
"""

import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd
from src.evaluator import RAGEvaluator


class TestRAGEvaluatorInit:
    """Tests for RAGEvaluator initialization."""

    def test_init_without_pipeline(self):
        """Test initialization without pipeline."""
        evaluator = RAGEvaluator()

        assert evaluator.pipeline is None
        assert evaluator.test_questions == []
        assert evaluator.evaluation_results == []

    def test_init_with_pipeline(self):
        """Test initialization with pipeline."""
        mock_pipeline = Mock()
        evaluator = RAGEvaluator(pipeline=mock_pipeline)

        assert evaluator.pipeline == mock_pipeline


class TestSetPipeline:
    """Tests for set_pipeline method."""

    def test_set_pipeline(self):
        """Test setting pipeline."""
        evaluator = RAGEvaluator()
        mock_pipeline = Mock()

        evaluator.set_pipeline(mock_pipeline)

        assert evaluator.pipeline == mock_pipeline


class TestCreateTestQuestions:
    """Tests for create_test_questions method."""

    def test_create_test_questions_custom(self):
        """Test creating custom test questions."""
        evaluator = RAGEvaluator()

        questions = [
            "Question 1?",
            "Question 2?",
            "Question 3?",
        ]
        categories = ["Cat1", "Cat2", "Cat3"]

        evaluator.create_test_questions(questions, categories)

        assert len(evaluator.test_questions) == 3
        assert evaluator.test_questions[0]["question"] == "Question 1?"
        assert evaluator.test_questions[0]["category"] == "Cat1"
        assert evaluator.test_questions[0]["id"] == 1

    def test_create_test_questions_default(self):
        """Test creating default test questions."""
        evaluator = RAGEvaluator()

        evaluator.create_test_questions()

        assert len(evaluator.test_questions) > 0
        assert all("question" in q for q in evaluator.test_questions)
        assert all("category" in q for q in evaluator.test_questions)

    def test_create_test_questions_empty(self):
        """Test creating test questions with empty list."""
        evaluator = RAGEvaluator()

        with pytest.raises(ValueError, match="questions must be a non-empty list"):
            evaluator.create_test_questions(questions=[])

    def test_create_test_questions_with_expected_answers(self):
        """Test creating questions with expected answers."""
        evaluator = RAGEvaluator()

        questions = ["Q1?", "Q2?"]
        expected_answers = ["A1", "A2"]

        evaluator.create_test_questions(questions, expected_answers=expected_answers)

        assert evaluator.test_questions[0]["expected_answer"] == "A1"
        assert evaluator.test_questions[1]["expected_answer"] == "A2"


class TestEvaluateResponse:
    """Tests for evaluate_response method."""

    def test_evaluate_response_basic(self):
        """Test basic response evaluation."""
        evaluator = RAGEvaluator()

        question = "What are the issues?"
        response = "The main issues are fees and poor service quality."
        sources = [
            {
                "chunk_text": "Complaint about fees",
                "metadata": {"product": "Credit card"},
            },
            {"chunk_text": "Poor service", "metadata": {"product": "Personal loan"}},
        ]

        evaluation = evaluator.evaluate_response(question, response, sources)

        assert "question" in evaluation
        assert "response" in evaluation
        assert "scores" in evaluation
        assert "overall_score" in evaluation
        assert evaluation["num_sources"] == 2
        assert evaluation["overall_score"] > 0

    def test_evaluate_response_no_sources(self):
        """Test evaluation with no sources."""
        evaluator = RAGEvaluator()

        evaluation = evaluator.evaluate_response("Q?", "A", [])

        assert evaluation["num_sources"] == 0
        assert evaluation["scores"]["accuracy"] == 1  # Low score for no sources

    def test_evaluate_response_short_response(self):
        """Test evaluation with short response."""
        evaluator = RAGEvaluator()

        evaluation = evaluator.evaluate_response(
            "Q?", "Short", [{"chunk_text": "Source"}]
        )

        assert evaluation["scores"]["relevance"] == 1  # Low score for short response

    def test_evaluate_response_comprehensive(self):
        """Test evaluation with comprehensive response."""
        evaluator = RAGEvaluator()

        long_response = (
            "This is a very detailed and comprehensive response that covers multiple aspects of the question. "
            * 5
        )
        sources = [
            {"chunk_text": f"Source {i}", "metadata": {"product": "Credit card"}}
            for i in range(5)
        ]

        evaluation = evaluator.evaluate_response("Q?", long_response, sources)

        assert evaluation["scores"]["completeness"] >= 3
        assert evaluation["scores"]["accuracy"] >= 3


class TestRunEvaluation:
    """Tests for run_evaluation method."""

    def test_run_evaluation_without_pipeline(self):
        """Test evaluation without pipeline set."""
        evaluator = RAGEvaluator()

        with pytest.raises(ValueError, match="Pipeline must be set"):
            evaluator.run_evaluation()

    def test_run_evaluation_without_questions(self):
        """Test evaluation without questions."""
        evaluator = RAGEvaluator()
        evaluator.set_pipeline(Mock())

        with pytest.raises(ValueError, match="No questions available"):
            evaluator.run_evaluation()

    def test_run_evaluation_with_test_questions(self):
        """Test evaluation with predefined test questions."""
        evaluator = RAGEvaluator()

        # Create mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.query.return_value = "Generated response"
        mock_pipeline.get_sources.return_value = [
            {"chunk_text": "Source", "metadata": {"product": "Credit card"}}
        ]

        evaluator.set_pipeline(mock_pipeline)
        evaluator.create_test_questions(questions=["Q1?", "Q2?"])

        df = evaluator.run_evaluation()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "Question ID" in df.columns
        assert "Overall Score" in df.columns
        assert len(evaluator.evaluation_results) == 2

    def test_run_evaluation_with_custom_questions(self):
        """Test evaluation with custom questions."""
        evaluator = RAGEvaluator()

        mock_pipeline = Mock()
        mock_pipeline.query.return_value = "Answer"
        mock_pipeline.get_sources.return_value = []

        evaluator.set_pipeline(mock_pipeline)

        df = evaluator.run_evaluation(questions=["Custom Q1?", "Custom Q2?"])

        assert len(df) == 2

    def test_run_evaluation_with_error(self):
        """Test evaluation handling errors."""
        evaluator = RAGEvaluator()

        mock_pipeline = Mock()
        mock_pipeline.query.side_effect = Exception("Query failed")

        evaluator.set_pipeline(mock_pipeline)
        evaluator.create_test_questions(questions=["Q1?"])

        df = evaluator.run_evaluation()

        assert len(df) == 1
        assert evaluator.evaluation_results[0]["success"] is False
        assert "error" in evaluator.evaluation_results[0]


class TestGenerateReport:
    """Tests for generate_report method."""

    def test_generate_report_without_results(self):
        """Test generating report without results."""
        evaluator = RAGEvaluator()

        with pytest.raises(ValueError, match="No evaluation results"):
            evaluator.generate_report()

    def test_generate_markdown_report(self):
        """Test generating markdown report."""
        evaluator = RAGEvaluator()

        # Add mock results
        evaluator.evaluation_results = [
            {
                "question_id": 1,
                "category": "Test",
                "question": "Q1?",
                "response": "A1",
                "num_sources": 2,
                "overall_score": 4.0,
                "scores": {"relevance": 4, "completeness": 4},
                "comments": ["Good response"],
                "success": True,
            }
        ]

        report = evaluator.generate_report(output_format="markdown")

        assert isinstance(report, str)
        assert "# RAG Pipeline Evaluation Report" in report
        assert "Summary" in report
        assert "Q1?" in report

    def test_generate_text_report(self):
        """Test generating text report."""
        evaluator = RAGEvaluator()

        evaluator.evaluation_results = [
            {
                "question_id": 1,
                "category": "Test",
                "question": "Q?",
                "response": "A",
                "num_sources": 1,
                "overall_score": 3.0,
                "scores": {},
                "comments": [],
                "success": True,
            }
        ]

        report = evaluator.generate_report(output_format="text")

        assert isinstance(report, str)
        assert "Q?" in report

    def test_generate_html_report(self):
        """Test generating HTML report."""
        evaluator = RAGEvaluator()

        evaluator.evaluation_results = [
            {
                "question_id": 1,
                "category": "Test",
                "question": "Q?",
                "response": "A",
                "num_sources": 1,
                "overall_score": 3.5,
                "scores": {},
                "comments": [],
                "success": True,
            }
        ]

        report = evaluator.generate_report(output_format="html")

        assert isinstance(report, str)
        assert "<html>" in report
        assert "<body>" in report
        assert "Q?" in report

    def test_generate_report_invalid_format(self):
        """Test generating report with invalid format."""
        evaluator = RAGEvaluator()

        evaluator.evaluation_results = [
            {
                "question_id": 1,
                "success": True,
                "question": "Q?",
                "response": "A",
                "overall_score": 3.0,
            }
        ]

        with pytest.raises(ValueError, match="output_format must be"):
            evaluator.generate_report(output_format="invalid")


class TestGetStatistics:
    """Tests for get_statistics method."""

    def test_get_statistics_no_results(self):
        """Test statistics without results."""
        evaluator = RAGEvaluator()

        stats = evaluator.get_statistics()

        assert "message" in stats

    def test_get_statistics_with_results(self):
        """Test statistics with evaluation results."""
        evaluator = RAGEvaluator()

        evaluator.evaluation_results = [
            {
                "success": True,
                "overall_score": 4.0,
                "num_sources": 3,
                "category": "Product-specific",
            },
            {
                "success": True,
                "overall_score": 3.5,
                "num_sources": 2,
                "category": "Product-specific",
            },
            {
                "success": True,
                "overall_score": 4.5,
                "num_sources": 4,
                "category": "Cross-product",
            },
        ]

        stats = evaluator.get_statistics()

        assert stats["total_questions"] == 3
        assert stats["successful"] == 3
        assert stats["failed"] == 0
        assert stats["average_score"] == 4.0
        assert stats["max_score"] == 4.5
        assert stats["min_score"] == 3.5
        assert "category_scores" in stats

    def test_get_statistics_with_failures(self):
        """Test statistics with some failed evaluations."""
        evaluator = RAGEvaluator()

        evaluator.evaluation_results = [
            {
                "success": True,
                "overall_score": 4.0,
                "num_sources": 2,
                "category": "Test",
            },
            {"success": False, "error": "Error"},
        ]

        stats = evaluator.get_statistics()

        assert stats["total_questions"] == 2
        assert stats["successful"] == 1
        assert stats["failed"] == 1


class TestRAGEvaluatorIntegration:
    """Integration tests for RAGEvaluator."""

    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        # Create mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.query.side_effect = [
            "Answer to question 1 with sufficient detail",
            "Answer to question 2 with sufficient detail",
            "Answer to question 3 with sufficient detail",
        ]
        mock_pipeline.get_sources.return_value = [
            {
                "chunk_text": "Source text",
                "metadata": {"product": "Credit card", "complaint_id": "123"},
            },
            {
                "chunk_text": "Another source",
                "metadata": {"product": "Personal loan", "complaint_id": "456"},
            },
        ]

        # Create evaluator
        evaluator = RAGEvaluator(pipeline=mock_pipeline)

        # Create test questions
        questions = [
            "What are credit card issues?",
            "What are loan problems?",
            "What are common complaints?",
        ]
        categories = ["Product", "Product", "General"]

        evaluator.create_test_questions(questions, categories)

        # Run evaluation
        df = evaluator.run_evaluation()

        # Verify results
        assert len(df) == 3
        assert all(df["Overall Score"] > 0)

        # Get statistics
        stats = evaluator.get_statistics()
        assert stats["total_questions"] == 3
        assert stats["successful"] == 3

        # Generate report
        report = evaluator.generate_report(output_format="markdown")
        assert "Question 1" in report
        assert "Question 2" in report
        assert "Question 3" in report
