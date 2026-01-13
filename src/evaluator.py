"""
Evaluator Module for RAG Complaint Chatbot.

This module provides functionality to evaluate the quality and performance
of the RAG pipeline using predefined test questions and quality metrics.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
from src.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluates RAG pipeline performance using test questions and quality metrics.

    This class handles creating evaluation questions, running evaluations,
    assessing response quality, and generating comprehensive evaluation reports.

    Attributes:
        pipeline (RAGPipeline): The RAG pipeline to evaluate.
        test_questions (List[Dict]): List of test question dictionaries.
        evaluation_results (List[Dict]): Results from evaluations.
    """

    def __init__(self, pipeline: Optional[RAGPipeline] = None):
        """
        Initialize the RAGEvaluator.

        Args:
            pipeline: RAGPipeline instance to evaluate. Can be set later.
        """
        self.pipeline = pipeline
        self.test_questions = []
        self.evaluation_results = []

        logger.info("RAGEvaluator initialized")

    def set_pipeline(self, pipeline: RAGPipeline) -> None:
        """
        Set or update the RAG pipeline to evaluate.

        Args:
            pipeline: RAGPipeline instance.
        """
        self.pipeline = pipeline
        logger.info("Pipeline set for evaluation")

    def create_test_questions(
        self,
        questions: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        expected_answers: Optional[List[str]] = None,
    ) -> None:
        """
        Create or load test questions for evaluation.

        Args:
            questions: List of question strings.
            categories: Optional list of categories for each question.
            expected_answers: Optional list of expected/ideal answers.

        Raises:
            ValueError: If questions list is empty.
        """
        if questions is None:
            # Use default test questions
            questions = self._get_default_questions()
            categories = self._get_default_categories()

        if not questions:
            raise ValueError("questions must be a non-empty list")

        # Create question dictionaries
        self.test_questions = []

        for i, question in enumerate(questions):
            question_dict = {
                "id": i + 1,
                "question": question,
                "category": categories[i] if categories and i < len(categories) else "General",
                "expected_answer": (
                    expected_answers[i]
                    if expected_answers and i < len(expected_answers)
                    else None
                ),
            }
            self.test_questions.append(question_dict)

        logger.info(f"Created {len(self.test_questions)} test questions")

    def _get_default_questions(self) -> List[str]:
        """
        Get default set of test questions for evaluation.

        Returns:
            List of default test questions.
        """
        return [
            "What are the main issues customers face with credit cards?",
            "Why are people unhappy with personal loans?",
            "What complaints do customers have about savings accounts?",
            "What problems do users experience with money transfers?",
            "Which product has the most complaints about fees?",
            "What are common issues across all financial products?",
            "How do customers describe service quality problems?",
            "What are the most frequent billing-related complaints?",
            "What issues do customers report about account access?",
            "What concerns do customers raise about fraud or security?",
        ]

    def _get_default_categories(self) -> List[str]:
        """
        Get categories for default questions.

        Returns:
            List of categories corresponding to default questions.
        """
        return [
            "Product-specific",
            "Product-specific",
            "Product-specific",
            "Product-specific",
            "Cross-product",
            "Cross-product",
            "Issue-specific",
            "Issue-specific",
            "Issue-specific",
            "Issue-specific",
        ]

    def evaluate_response(
        self,
        question: str,
        response: str,
        sources: List[Dict[str, Any]],
        criteria: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a single response.

        Args:
            question: The question asked.
            response: The generated response.
            sources: Retrieved source documents.
            criteria: Optional list of evaluation criteria.

        Returns:
            Dictionary with evaluation scores and analysis.
        """
        if criteria is None:
            criteria = [
                "relevance",
                "completeness",
                "accuracy",
                "clarity",
                "source_usage",
            ]

        evaluation = {
            "question": question,
            "response": response,
            "num_sources": len(sources),
            "response_length": len(response),
            "scores": {},
            "overall_score": 0.0,
            "comments": [],
        }

        # Evaluate based on criteria
        for criterion in criteria:
            score, comment = self._evaluate_criterion(
                criterion, question, response, sources
            )
            evaluation["scores"][criterion] = score
            if comment:
                evaluation["comments"].append(f"{criterion.title()}: {comment}")

        # Calculate overall score
        evaluation["overall_score"] = sum(evaluation["scores"].values()) / len(
            evaluation["scores"]
        )

        return evaluation

    def _evaluate_criterion(
        self,
        criterion: str,
        question: str,
        response: str,
        sources: List[Dict[str, Any]],
    ) -> Tuple[int, str]:
        """
        Evaluate a single criterion (heuristic-based).

        Args:
            criterion: Name of the criterion to evaluate.
            question: The question.
            response: The generated response.
            sources: Retrieved sources.

        Returns:
            Tuple of (score 1-5, comment).
        """
        score = 3  # Default neutral score
        comment = ""

        if criterion == "relevance":
            # Check if response addresses the question
            if len(response) < 20:
                score = 1
                comment = "Response too short"
            elif "don't have enough information" in response.lower():
                score = 2
                comment = "Unable to answer from context"
            else:
                score = 4
                comment = "Response addresses the question"

        elif criterion == "completeness":
            # Check response length and detail
            if len(response) < 50:
                score = 2
                comment = "Response lacks detail"
            elif len(response) > 200:
                score = 4
                comment = "Comprehensive response"
            else:
                score = 3
                comment = "Adequate detail"

        elif criterion == "accuracy":
            # Check if response seems grounded in sources
            if len(sources) == 0:
                score = 1
                comment = "No sources retrieved"
            elif len(sources) >= 3:
                score = 4
                comment = f"Based on {len(sources)} sources"
            else:
                score = 3
                comment = f"Based on {len(sources)} source(s)"

        elif criterion == "clarity":
            # Simple heuristic for clarity
            if len(response) > 500:
                score = 3
                comment = "Response may be too verbose"
            else:
                score = 4
                comment = "Clear and concise"

        elif criterion == "source_usage":
            # Check if response mentions specific details from sources
            if len(sources) == 0:
                score = 1
                comment = "No sources available"
            else:
                # Check if response references products/issues from sources
                has_specifics = any(
                    source.get("metadata", {}).get("product", "").lower() in response.lower()
                    for source in sources
                )
                if has_specifics:
                    score = 4
                    comment = "Response uses source details"
                else:
                    score = 3
                    comment = "Could better utilize sources"

        return score, comment

    def run_evaluation(
        self,
        questions: Optional[List[str]] = None,
        save_results: bool = True,
    ) -> pd.DataFrame:
        """
        Run evaluation on test questions.

        Args:
            questions: Optional list of questions (uses test_questions if None).
            save_results: Whether to save results to evaluation_results.

        Returns:
            DataFrame with evaluation results.

        Raises:
            ValueError: If pipeline is not set or questions are not available.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be set before running evaluation")

        # Determine which questions to use
        if questions is not None:
            eval_questions = [
                {"id": i + 1, "question": q, "category": "Custom"}
                for i, q in enumerate(questions)
            ]
        elif self.test_questions:
            eval_questions = self.test_questions
        else:
            raise ValueError(
                "No questions available. Call create_test_questions() first."
            )

        logger.info(f"Running evaluation on {len(eval_questions)} questions")

        results = []

        for q_data in eval_questions:
            question = q_data["question"]
            q_id = q_data["id"]
            category = q_data.get("category", "General")

            logger.info(f"Evaluating question {q_id}: {question[:50]}...")

            try:
                # Query the pipeline
                response = self.pipeline.query(question)
                sources = self.pipeline.get_sources()

                # Evaluate the response
                evaluation = self.evaluate_response(question, response, sources)

                # Add metadata
                evaluation["question_id"] = q_id
                evaluation["category"] = category
                evaluation["success"] = True
                evaluation["error"] = None

                results.append(evaluation)

            except Exception as e:
                logger.error(f"Error evaluating question {q_id}: {e}")
                results.append(
                    {
                        "question_id": q_id,
                        "question": question,
                        "category": category,
                        "success": False,
                        "error": str(e),
                        "overall_score": 0.0,
                    }
                )

        # Save results
        if save_results:
            self.evaluation_results = results

        # Create DataFrame
        df = self._create_results_dataframe(results)

        logger.info("Evaluation completed")
        return df

    def _create_results_dataframe(
        self, results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from evaluation results.

        Args:
            results: List of evaluation result dictionaries.

        Returns:
            DataFrame with formatted results.
        """
        rows = []

        for result in results:
            if result.get("success", False):
                row = {
                    "Question ID": result["question_id"],
                    "Category": result["category"],
                    "Question": result["question"][:80] + "..."
                    if len(result["question"]) > 80
                    else result["question"],
                    "Response": result["response"][:100] + "..."
                    if len(result["response"]) > 100
                    else result["response"],
                    "Num Sources": result["num_sources"],
                    "Overall Score": round(result["overall_score"], 2),
                }

                # Add individual scores
                for criterion, score in result.get("scores", {}).items():
                    row[criterion.title()] = score

                row["Comments"] = "; ".join(result.get("comments", []))

                rows.append(row)
            else:
                # Failed evaluation
                rows.append(
                    {
                        "Question ID": result["question_id"],
                        "Category": result["category"],
                        "Question": result["question"],
                        "Response": f"ERROR: {result.get('error', 'Unknown error')}",
                        "Overall Score": 0.0,
                    }
                )

        return pd.DataFrame(rows)

    def generate_report(
        self,
        output_format: str = "markdown",
        include_details: bool = True,
    ) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            output_format: Format for the report ('markdown', 'text', or 'html').
            include_details: Whether to include detailed response/source info.

        Returns:
            Formatted report string.

        Raises:
            ValueError: If no evaluation results are available.
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results. Run run_evaluation() first.")

        if output_format == "markdown":
            return self._generate_markdown_report(include_details)
        elif output_format == "text":
            return self._generate_text_report(include_details)
        elif output_format == "html":
            return self._generate_html_report(include_details)
        else:
            raise ValueError("output_format must be 'markdown', 'text', or 'html'")

    def _generate_markdown_report(self, include_details: bool = True) -> str:
        """Generate markdown-formatted report."""
        report = ["# RAG Pipeline Evaluation Report\n"]
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Summary statistics
        successful = [r for r in self.evaluation_results if r.get("success", False)]
        if successful:
            avg_score = sum(r["overall_score"] for r in successful) / len(successful)
            report.append("## Summary\n")
            report.append(f"- **Total Questions:** {len(self.evaluation_results)}")
            report.append(f"- **Successful:** {len(successful)}")
            report.append(f"- **Failed:** {len(self.evaluation_results) - len(successful)}")
            report.append(f"- **Average Score:** {avg_score:.2f}/5.0\n")

        # Detailed results
        report.append("## Evaluation Results\n")

        for i, result in enumerate(self.evaluation_results, 1):
            if result.get("success", False):
                report.append(f"### Question {result['question_id']}: {result['category']}\n")
                report.append(f"**Question:** {result['question']}\n")
                report.append(
                    f"**Overall Score:** {result['overall_score']:.2f}/5.0\n"
                )

                # Individual scores
                report.append("**Scores:**")
                for criterion, score in result.get("scores", {}).items():
                    report.append(f"- {criterion.title()}: {score}/5")
                report.append("")

                if include_details:
                    report.append(f"**Response:** {result['response']}\n")
                    report.append(f"**Sources Used:** {result['num_sources']}\n")

                # Comments
                if result.get("comments"):
                    report.append("**Analysis:**")
                    for comment in result["comments"]:
                        report.append(f"- {comment}")
                    report.append("")

                report.append("---\n")

        return "\n".join(report)

    def _generate_text_report(self, include_details: bool = True) -> str:
        """Generate plain text report."""
        # Convert markdown to simpler text format
        md_report = self._generate_markdown_report(include_details)
        text_report = md_report.replace("#", "").replace("**", "").replace("---", "=" * 50)
        return text_report

    def _generate_html_report(self, include_details: bool = True) -> str:
        """Generate HTML report."""
        html = ["<html><head><style>"]
        html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("h1 { color: #333; }")
        html.append("h2 { color: #666; margin-top: 30px; }")
        html.append(".result { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }")
        html.append(".score { font-weight: bold; color: #2196F3; }")
        html.append("</style></head><body>")

        html.append("<h1>RAG Pipeline Evaluation Report</h1>")
        html.append(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")

        # Summary
        successful = [r for r in self.evaluation_results if r.get("success", False)]
        if successful:
            avg_score = sum(r["overall_score"] for r in successful) / len(successful)
            html.append("<h2>Summary</h2>")
            html.append(f"<p><strong>Average Score:</strong> <span class='score'>{avg_score:.2f}/5.0</span></p>")

        # Results
        html.append("<h2>Results</h2>")
        for result in self.evaluation_results:
            if result.get("success", False):
                html.append("<div class='result'>")
                html.append(f"<h3>Question {result['question_id']}: {result['category']}</h3>")
                html.append(f"<p><strong>Q:</strong> {result['question']}</p>")
                html.append(
                    f"<p><strong>Score:</strong> <span class='score'>{result['overall_score']:.2f}/5.0</span></p>"
                )

                if include_details:
                    html.append(f"<p><strong>Response:</strong> {result['response']}</p>")

                html.append("</div>")

        html.append("</body></html>")
        return "".join(html)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get evaluation statistics.

        Returns:
            Dictionary with statistics.
        """
        if not self.evaluation_results:
            return {"message": "No evaluation results available"}

        successful = [r for r in self.evaluation_results if r.get("success", False)]

        if not successful:
            return {"message": "No successful evaluations"}

        stats = {
            "total_questions": len(self.evaluation_results),
            "successful": len(successful),
            "failed": len(self.evaluation_results) - len(successful),
            "average_score": sum(r["overall_score"] for r in successful) / len(successful),
            "max_score": max(r["overall_score"] for r in successful),
            "min_score": min(r["overall_score"] for r in successful),
            "average_sources": sum(r["num_sources"] for r in successful) / len(successful),
        }

        # Score distribution by category
        categories = {}
        for result in successful:
            cat = result.get("category", "Unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result["overall_score"])

        stats["category_scores"] = {
            cat: sum(scores) / len(scores) for cat, scores in categories.items()
        }

        return stats

    def __repr__(self) -> str:
        """String representation of the RAGEvaluator."""
        return (
            f"RAGEvaluator(test_questions={len(self.test_questions)}, "
            f"results={len(self.evaluation_results)})"
        )
