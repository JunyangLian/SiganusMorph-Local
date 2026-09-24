#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/home/admin/yangzz/SiganusMorph}"
PAGES_DIR="${PROJECT_ROOT}/pages"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="${PROJECT_ROOT}/developer_tools/legacy_streamlit_pages/${STAMP}"

if [[ ! -f "${PROJECT_ROOT}/app.py" || ! -d "${PAGES_DIR}" ]]; then
  echo "Invalid SiganusMorph project root: ${PROJECT_ROOT}" >&2
  exit 1
fi

mkdir -p "${ARCHIVE_DIR}"

for path in "${PAGES_DIR}"/*.py; do
  [[ -e "${path}" ]] || continue
  name="$(basename "${path}")"
  case "${name}" in
    1_single_fish.py|2_batch_measurement.py|3_results_export.py|4_contact.py)
      echo "public: ${name}"
      ;;
    *)
      echo "archive: ${name}"
      mv -- "${path}" "${ARCHIVE_DIR}/${name}"
      ;;
  esac
done

echo "Legacy pages preserved at: ${ARCHIVE_DIR}"
