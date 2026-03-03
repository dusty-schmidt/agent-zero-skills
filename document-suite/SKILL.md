---
name: document-suite
description: Comprehensive office document handling for Word, PDF, PowerPoint, and Excel files. Use for creating, reading, editing, converting, and manipulating documents in any .docx, .pdf, .pptx, .xlsx, .csv, or .tsv format. Covers presentations, spreadsheets, word processing, PDF extraction, form filling, merging, and format conversion.
license: Proprietary
tags: [docx, pdf, pptx, xlsx, office, documents, spreadsheets, presentations, word, excel]
author: GOB
version: 3.0.0
---

# Document Suite

Unified skill for all office document operations.

## Supported Formats

| Format | Extension | Operations |
|--------|-----------|------------|
| Word | .docx | Create, edit, format, convert |
| PDF | .pdf | Create, extract, merge, split, OCR, forms |
| PowerPoint | .pptx | Create, edit, extract, merge |
| Excel | .xlsx, .xlsm | Create, edit, formulas, charts, pivot |
| CSV/TSV | .csv, .tsv | Read, edit, convert |

## Use Cases

### Word Documents (.docx)
- Generate reports with tables of contents
- Create professional documents with letterheads
- Perform find/replace operations
- Work with tracked changes and comments
- Convert to/from other formats

### PDF Operations
- Extract text and tables from PDFs
- Merge multiple PDFs into one
- Split PDFs into separate files
- Rotate pages, add watermarks
- Fill PDF forms programmatically
- OCR on scanned PDFs
- Encrypt/decrypt PDFs

### Presentations (.pptx)
- Create slide decks and pitch decks
- Extract text from existing presentations
- Edit and modify slides
- Work with templates and layouts
- Merge or split slide files
- Handle speaker notes and comments

### Spreadsheets (.xlsx, .csv)
- Create new spreadsheets from data
- Read and edit existing files
- Compute formulas and charts
- Clean messy data
- Format cells, add conditional formatting
- Create pivot tables
- Convert between formats

## Quick Reference

| Task | Tool |
|------|------|
| Create Word doc | python-docx |
| Edit PDF | PyPDF2 / pdfplumber |
| Create presentation | python-pptx |
| Edit Excel | openpyxl / pandas |
| OCR PDF | pytesseract |
| Merge PDFs | PyPDF2 |
| CSV analysis | pandas |
