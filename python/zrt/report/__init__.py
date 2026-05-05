"""zrt.report — Performance reporting."""
from python.zrt.report.summary import (
    E2ESummary, build_summary,
    TrainingSummary, build_training_summary,
)
from python.zrt.report.html_writer import (
    export_html_report, export_hierarchical_html_report, export_reports,
)
from python.zrt.report.chrome_trace import (
    build_chrome_trace, build_chrome_trace_multi,
    export_chrome_trace, export_chrome_trace_multi,
)
from python.zrt.report.compare import (
    ComparisonReport, build_comparison_report,
    export_comparison_excel, export_comparison_html,
)

# Phase 1: hierarchical report types
from python.zrt.report.report_types import (
    ReportContext, BlockDetail, SubStructureDetail,
    OpFamilyDetail, OpDetail,
)
from python.zrt.report.report_builder import build_report_context
from python.zrt.report.formula_registry import FormulaRegistry, FormulaEntry


def get_formula_registry() -> FormulaRegistry:
    """Factory: lazily returns the singleton FormulaRegistry instance."""
    return FormulaRegistry()
from python.zrt.report.shape_desc import describe_shapes
from python.zrt.report.topology_renderer import render_topology_svg
from python.zrt.report.structure_renderer import render_structure_svg, render_structure_html
from python.zrt.report.dot_exporter import export_dot, render_dot

__all__ = [
    "E2ESummary", "build_summary",
    "TrainingSummary", "build_training_summary",
    "export_html_report", "export_hierarchical_html_report", "export_reports",
    "build_chrome_trace", "build_chrome_trace_multi",
    "export_chrome_trace", "export_chrome_trace_multi",
    "ComparisonReport", "build_comparison_report",
    "export_comparison_excel", "export_comparison_html",
    # Phase 1 types
    "ReportContext", "BlockDetail", "SubStructureDetail",
    "OpFamilyDetail", "OpDetail",
    "build_report_context",
    "get_formula_registry", "FormulaRegistry", "FormulaEntry",
    "describe_shapes",
    "render_topology_svg", "render_structure_svg", "render_structure_html",
    "export_dot", "render_dot",
]
