from pathlib import Path


def test_launcher_dp_formula_excludes_ep_from_rank_product():
    html = Path("server/launcher.html").read_text(encoding="utf-8")

    assert "tp * pp * ep * cp" not in html
    assert "const denom = tp * pp * cp;" in html
    assert "EP shares ranks" in html


def test_launcher_file_protocol_defaults_to_localhost_api():
    html = Path("server/launcher.html").read_text(encoding="utf-8")

    assert "location.protocol === 'file:'" in html
    assert "http://localhost:8000" in html


def test_launcher_export_buttons_have_visible_click_handlers():
    html = Path("server/launcher.html").read_text(encoding="utf-8")

    assert 'data-artifact-kind="excel"' in html
    assert 'data-artifact-kind="html"' not in html
    assert "Open HTML" not in html
    assert "document.addEventListener('click'" in html
    assert "/open" not in html
    assert "downloadArtifact(jobId, kind)" in html
    assert "window.open(url, '_blank', 'noopener')" not in html
    assert "setActionMessage(jobId" in html


def test_artifact_routes_do_not_launch_apps_on_server_host():
    from server.main import app

    assert all(not route.path.endswith("/open") for route in app.routes)


def test_html_artifact_is_served_inline_for_embedded_report(tmp_path):
    from server.main import _jobs, _lock, get_job_artifact

    html_artifact = tmp_path / "report.html"
    html_artifact.write_text("<!doctype html><title>report</title>", encoding="utf-8")
    excel_artifact = tmp_path / "report.xlsx"
    excel_artifact.write_bytes(b"xlsx")
    job_id = "inline-report-test"

    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": "done",
            "result": {
                "html_filename": html_artifact.name,
                "html_path": str(html_artifact),
                "excel_filename": excel_artifact.name,
                "excel_path": str(excel_artifact),
            },
            "error": None,
            "created_at": "now",
            "finished_at": "now",
        }

    html_response = get_job_artifact(job_id, html_artifact.name)
    excel_response = get_job_artifact(job_id, excel_artifact.name)

    assert html_response.media_type == "text/html"
    assert "attachment" not in html_response.headers.get("content-disposition", "").lower()
    assert excel_response.headers["content-disposition"].lower().startswith("attachment")


def test_launcher_report_uses_url_parser_instead_of_string_concat():
    html = Path("server/launcher.html").read_text(encoding="utf-8")

    assert "function apiUrl(path)" in html
    assert "new URL(path, `${apiBase()}/`).href" in html
    assert "loadReportFrame(jobId, apiUrl(url));" in html
    assert 'src="${api()}${url}"' not in html


def test_launcher_report_has_loading_progress_overlay():
    html = Path("server/launcher.html").read_text(encoding="utf-8")

    assert "report-loader" in html
    assert "report-progress-bar" in html
    assert "async function loadReportFrame(jobId, url)" in html
    assert "function updateReportProgress(jobId, percent, msg, indeterminate=false)" in html
    assert "Content-Length" in html
    assert "reader.read()" in html


def test_launcher_disables_form_estimate_until_catalogs_load():
    html = Path("server/launcher.html").read_text(encoding="utf-8")

    assert 'id="btn-est" disabled' in html
    assert "let formModelsReady = false;" in html
    assert "let formHardwareReady = false;" in html
    assert "function formCatalogReady()" in html
    assert "if (isEstimateFormMode() && !formCatalogReady())" in html
    assert "setEstimateRunState();" in html


def test_estimate_job_returns_html_and_excel_artifacts(tmp_path):
    from server.main import EstimateRequest, _do_estimate

    result = _do_estimate(
        EstimateRequest(
            config_path="python/zrt/training/configs/llama3_70b_3d.yaml",
            output_dir=str(tmp_path),
        ),
        job_id="test-job",
    )

    assert result["html_filename"].endswith(".html")
    assert result["excel_filename"].endswith(".xlsx")
    assert result["html_url"].endswith(result["html_filename"])
    assert result["excel_url"].endswith(result["excel_filename"])
    assert (tmp_path / result["html_filename"]).exists()
    assert (tmp_path / result["excel_filename"]).exists()
    assert "TP*CP*PP*DP" in result["data"]["config_summary"]["parallelism"]
    assert "TP*CP*PP*EP*DP" not in result["data"]["config_summary"]["parallelism"]
