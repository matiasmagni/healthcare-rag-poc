"""Generate standalone HTML report for Ragas evaluation results."""
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def generate_html_report(df: pd.DataFrame, result=None):
    """Generate a styled, standalone HTML report.

    Args:
        df: DataFrame containing evaluation results
        result: Ragas evaluation result object (optional)
    """
    df = df.copy()

    if "answer_relevance" not in df.columns and "answer_relevancy" in df.columns:
        df["answer_relevance"] = pd.to_numeric(df["answer_relevancy"], errors="coerce").fillna(0.0)

    def safe_series(column: str, default: float = 0.0) -> pd.Series:
        if column not in df.columns:
            return pd.Series([default] * len(df), index=df.index, dtype=float)
        return pd.to_numeric(df[column], errors="coerce").fillna(default)

    report_path_env = os.getenv("RAGAS_REPORT_PATH", "tests/reports/ragas_report.html")
    report_path = Path(report_path_env)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    reports_dir = report_path.parent
    reports_dir.mkdir(parents=True, exist_ok=True)

    faith_series = safe_series("faithfulness")
    answer_relevance_series = safe_series("answer_relevance")
    context_precision_series = safe_series("context_precision")
    context_recall_series = safe_series("context_recall")

    faith_avg = float(faith_series.mean())
    answer_rel_avg = float(answer_relevance_series.mean())
    context_prec_avg = float(context_precision_series.mean())
    context_rec_avg = float(context_recall_series.mean())

    thresholds = {
        "faithfulness": float(os.getenv("RAGAS_FAITHFULNESS_THRESHOLD", "0.85")),
        "answer_relevance": float(os.getenv("RAGAS_ANSWER_RELEVANCE_THRESHOLD", "0.85")),
        "context_precision": float(os.getenv("RAGAS_CONTEXT_PRECISION_THRESHOLD", "0.85")),
        "context_recall": float(os.getenv("RAGAS_CONTEXT_RECALL_THRESHOLD", "0.85")),
    }
    
    def get_badge_class(score, threshold):
        if score >= threshold:
            return "bg-green-100 text-green-800"
        return "bg-red-100 text-red-800"
    
    def get_badge_text(score, threshold):
        if score >= threshold:
            return "PASS"
        return "FAIL"
    
    # Build table rows
    table_rows = []
    for idx, row in df.iterrows():
        question = str(row.get("user_input", "")).replace("<", "&lt;").replace(">", "&gt;")
        answer = str(row.get("response", "")).replace("<", "&lt;").replace(">", "&gt;")
        reference = str(row.get("reference", "")).replace("<", "&lt;").replace(">", "&gt;")
        contexts = row.get("retrieved_contexts", [])
        if isinstance(contexts, list):
            contexts_str = "<br><br>".join(str(c).replace("<", "&lt;").replace(">", "&gt;") for c in contexts)
        else:
            contexts_str = str(contexts).replace("<", "&lt;").replace(">", "&gt;")
        
        faith = float(row.get("faithfulness", 0) or 0)
        answer_rel = float(row.get("answer_relevance", 0) or 0)
        context_prec = float(row.get("context_precision", 0) or 0)
        context_rec = float(row.get("context_recall", 0) or 0)
        
        table_rows.append(f"""
        <tr class="border-b border-gray-200 hover:bg-gray-50">
            <td class="px-6 py-4 text-sm text-gray-900 max-w-xs">
                <div class="font-medium">{question}</div>
            </td>
            <td class="px-6 py-4 text-sm text-gray-700 max-w-md">
                <div class="max-h-32 overflow-y-auto">{answer}</div>
            </td>
            <td class="px-6 py-4 text-sm text-gray-700 max-w-md">
                <div class="max-h-32 overflow-y-auto">{reference}</div>
            </td>
            <td class="px-6 py-4 text-sm text-gray-600 max-w-md">
                <div class="max-h-40 overflow-y-auto text-xs">{contexts_str}</div>
            </td>
            <td class="px-6 py-4 text-center">
                <span class="px-2 py-1 rounded-full text-xs font-semibold {get_badge_class(faith, thresholds['faithfulness'])}">
                    {faith:.3f} {get_badge_text(faith, thresholds['faithfulness'])}
                </span>
            </td>
            <td class="px-6 py-4 text-center">
                <span class="px-2 py-1 rounded-full text-xs font-semibold {get_badge_class(answer_rel, thresholds['answer_relevance'])}">
                    {answer_rel:.3f} {get_badge_text(answer_rel, thresholds['answer_relevance'])}
                </span>
            </td>
            <td class="px-6 py-4 text-center">
                <span class="px-2 py-1 rounded-full text-xs font-semibold {get_badge_class(context_prec, thresholds['context_precision'])}">
                    {context_prec:.3f} {get_badge_text(context_prec, thresholds['context_precision'])}
                </span>
            </td>
            <td class="px-6 py-4 text-center">
                <span class="px-2 py-1 rounded-full text-xs font-semibold {get_badge_class(context_rec, thresholds['context_recall'])}">
                    {context_rec:.3f} {get_badge_text(context_rec, thresholds['context_recall'])}
                </span>
            </td>
        </tr>
        """)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAGAS Evaluation Report - Healthcare RAG POC</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        }}
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="container mx-auto px-4 py-8 max-w-7xl">
        <!-- Header -->
        <div class="mb-8">
            <h1 class="text-4xl font-bold text-gray-900 mb-2">RAGAS Evaluation Report</h1>
            <p class="text-gray-600">Healthcare RAG POC - Automated Evaluation Results</p>
            <p class="text-sm text-gray-500 mt-1">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <!-- Summary Cards -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow-md p-6 border-l-4 {'border-green-500' if faith_avg >= thresholds['faithfulness'] else 'border-red-500'}">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600 uppercase tracking-wide">Faithfulness</p>
                        <p class="text-3xl font-bold text-gray-900 mt-2">{faith_avg:.3f}</p>
                        <p class="text-xs text-gray-500 mt-1">Threshold: {thresholds['faithfulness']}</p>
                    </div>
                    <div>
                        <span class="px-3 py-1 rounded-full text-sm font-semibold {get_badge_class(faith_avg, thresholds['faithfulness'])}">
                            {get_badge_text(faith_avg, thresholds['faithfulness'])}
                        </span>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow-md p-6 border-l-4 {'border-green-500' if answer_rel_avg >= thresholds['answer_relevance'] else 'border-red-500'}">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600 uppercase tracking-wide">Answer Relevance</p>
                        <p class="text-3xl font-bold text-gray-900 mt-2">{answer_rel_avg:.3f}</p>
                        <p class="text-xs text-gray-500 mt-1">Threshold: {thresholds['answer_relevance']}</p>
                    </div>
                    <div>
                        <span class="px-3 py-1 rounded-full text-sm font-semibold {get_badge_class(answer_rel_avg, thresholds['answer_relevance'])}">
                            {get_badge_text(answer_rel_avg, thresholds['answer_relevance'])}
                        </span>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow-md p-6 border-l-4 {'border-green-500' if context_prec_avg >= thresholds['context_precision'] else 'border-red-500'}">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600 uppercase tracking-wide">Context Precision</p>
                        <p class="text-3xl font-bold text-gray-900 mt-2">{context_prec_avg:.3f}</p>
                        <p class="text-xs text-gray-500 mt-1">Threshold: {thresholds['context_precision']}</p>
                    </div>
                    <div>
                        <span class="px-3 py-1 rounded-full text-sm font-semibold {get_badge_class(context_prec_avg, thresholds['context_precision'])}">
                            {get_badge_text(context_prec_avg, thresholds['context_precision'])}
                        </span>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow-md p-6 border-l-4 {'border-green-500' if context_rec_avg >= thresholds['context_recall'] else 'border-red-500'}">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600 uppercase tracking-wide">Context Recall</p>
                        <p class="text-3xl font-bold text-gray-900 mt-2">{context_rec_avg:.3f}</p>
                        <p class="text-xs text-gray-500 mt-1">Threshold: {thresholds['context_recall']}</p>
                    </div>
                    <div>
                        <span class="px-3 py-1 rounded-full text-sm font-semibold {get_badge_class(context_rec_avg, thresholds['context_recall'])}">
                            {get_badge_text(context_rec_avg, thresholds['context_recall'])}
                        </span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Detailed Results Table -->
        <div class="bg-white rounded-lg shadow-md overflow-hidden">
            <div class="px-6 py-4 bg-gray-100 border-b border-gray-200">
                <h2 class="text-xl font-bold text-gray-900">Detailed Results</h2>
                <p class="text-sm text-gray-600 mt-1">{len(df)} test case(s) evaluated</p>
            </div>
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Question</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Generated Answer</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Ground Truth</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Retrieved Contexts</th>
                            <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Faithfulness</th>
                            <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Answer Relevance</th>
                            <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Context Precision</th>
                            <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Context Recall</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                        {''.join(table_rows)}
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="mt-8 text-center text-sm text-gray-500">
            <p>RAGAS Evaluation Framework - Standalone HTML Report (No External Runtime Dependencies)</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return report_path


if __name__ == "__main__":
    # This module is intended to be imported by pytest and the test suite.
    pass
