import argparse
import re
from typing import Tuple, Optional

import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString


_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")

# ---- Meta patterns we want to remove (commonly at end of Tagesschau items) ----
# Strong/meta blocks like:
# "Über dieses Thema berichtete die tagesschau am 06. September 2025 um 12:00 Uhr."
# "Über dieses Thema berichtete Inforadio am 02. Juli 2025 um 07:24 Uhr."
# "Mit Informationen von ... "
# "Stand: ... "

_META_LINE_RE = re.compile(
    r"^(?:"
    r"Über\s+dieses\s+Thema\s+berichtete.*"
    r"|Mit\s+Informationen\s+von.*"
    r"|Stand\s*:\s*.*"
    r"|Anmerkung\s+der\s+Redaktion.*"
    r"|Hinweis\s*:\s*.*"
    r"|Korrektur\s*:\s*.*"
    r"|Dieser\s+(?:Beitrag|Artikel)\s+wurde\s+.*"
    r")$",
    re.IGNORECASE,
)

# Also remove trailing meta sentences if they survive into plain text
# (more permissive: can match within last ~250 chars)
_TRAILING_META_RE = re.compile(
    r"(?:\s*(?:"
    r"Über\s+dieses\s+Thema\s+berichtete.*?(?:\.\s*|$)"
    r"|Mit\s+Informationen\s+von.*?(?:\.\s*|$)"
    r"|Stand\s*:\s*.*?(?:\.\s*|$)"
    r"))\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_text(s: str) -> str:
    """
    Normalizes a text string by standardizing whitespace and punctuation spacing.

    Args: 
    - s (str): The input text string to be normalized

    Returns: 
    - str: Standardized text string 
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WHITESPACE_RE.sub(" ", s).strip()
    s = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)
    return s


def extract_subtitle_and_clean_html(html: str) -> Tuple[str, str]:
    """
    Returns (subtitle, cleaned_text_content)

    Rules:
    - If the FIRST non-whitespace top-level element is a <strong>, extract its text as subtitle,
      remove that <strong> from the content.
    - Remove <strong> blocks that are meta (typically at the end):
      e.g. "Über dieses Thema berichtete ...", "Mit Informationen von ...", "Stand: ..."
    - Remove <em> blocks that are meta:
        - empty after stripping, OR
        - contain any whitespace (multiple words / sentences / dates etc.)
      Keep <em> blocks with no whitespace (e.g., 'ARD-Morgenmagazin', 'tagesschau.de'),
      but unwrap them (keep text, drop tag).
    - Convert <a> to its inner text (BeautifulSoup text extraction does this naturally).
    - Strip all other tags to plain text.
    - Finally, strip trailing meta sentences from the resulting text.
    """
    if html is None:
        return "", ""

    html = str(html)

    # Parse as fragment inside a container to reliably inspect top-level nodes
    soup = BeautifulSoup(f"<div>{html}</div>", "html.parser")
    container = soup.div

    subtitle = ""

    # 1) Extract subtitle from first non-whitespace top-level node if it is <strong>
    first_meaningful = None
    for node in list(container.contents):
        if isinstance(node, NavigableString):
            if node.strip() == "":
                continue
            first_meaningful = node
            break
        if isinstance(node, Tag):
            if node.get_text(strip=True) == "":
                node.decompose()
                continue
            first_meaningful = node
            break

    if isinstance(first_meaningful, Tag) and first_meaningful.name == "strong":
        subtitle = first_meaningful.get_text(" ", strip=True)
        first_meaningful.decompose()

    # 2) Handle <em> tags according to your rules
    for em in container.find_all("em"):
        em_text = em.get_text(" ", strip=True)

        if em_text == "":
            em.decompose()
            continue

        # remove meta-ish <em> that contains any whitespace
        if re.search(r"\s", em_text):
            em.decompose()
            continue

        em.unwrap()

    # 3) Remove remaining <strong> that are meta (usually end-of-article hints)
    #    IMPORTANT: we already removed the first <strong> if it was subtitle
    for st in container.find_all("strong"):
        st_text = _normalize_text(st.get_text(" ", strip=True))
        if st_text and _META_LINE_RE.match(st_text):
            st.decompose()
        else:
            # If it's not meta, keep the text but drop formatting
            st.unwrap()

    # 4) Convert to plain text
    text = container.get_text(" ", strip=True)
    text = _normalize_text(text)

    subtitle = _normalize_text(subtitle)

    # 5) Remove trailing meta sentences that might still be present as plain text
    #    Apply repeatedly (sometimes multiple meta lines exist)
    prev = None
    while prev != text:
        prev = text
        text = _TRAILING_META_RE.sub("", text).strip()
        text = _normalize_text(text)

    return subtitle, text


def print_contents_per_label(df: pd.DataFrame, max_rows: int = 3) -> None:
    """
    Prints an overview of the dataframe columns

    Args:
    - df (pd.DataFrame): The dataframe 
    - max_rows (int): Maximum number of non-null example values printed per column, default is 3.

    Returns:
    - None.
    """
    print("\n=== CSV: per-label overview ===")
    for col in df.columns:
        non_null = df[col].notna().sum()
        print(f"\n[{col}]  dtype={df[col].dtype}  non-null={non_null}/{len(df)}")
        examples = df[col].dropna().head(max_rows).tolist()
        for i, ex in enumerate(examples, 1):
            ex_str = str(ex)
            if len(ex_str) > 300:
                ex_str = ex_str[:300] + "..."
            print(f"  ex{i}: {ex_str}")


def main() -> None:
    """
    Clean HTML from the news CSV, extract subtitles, and export a cleaned CSV.

    Args:
    - input_csv (str): Path to input CSV file
    - "-o", "--output_csv" (str): Path to output CSV (Default is "cleaned_export.csv")
    - "--no-print" (bool): If set, do not print contents per label
    - "--max-print-rows" (int): Number of example rows to print per label (Default is 3)

    Output:
    - Cleaned CSV file with the columns: 'index', 'title', 'subtitle', 'content', 'tags'
    """
    parser = argparse.ArgumentParser(
        description="Clean HTML from news CSV, extract subtitle from leading <strong>, drop meta strong lines, export reduced CSV."
    )
    parser.add_argument("input_csv", help="Path to input CSV")
    parser.add_argument(
        "-o",
        "--output_csv",
        default="cleaned_export.csv",
        help="Path to output CSV (default: cleaned_export.csv)",
    )
    parser.add_argument("--no-print", action="store_true", help="Do not print per-label overview")
    parser.add_argument("--max-print-rows", type=int, default=3, help="How many example rows to print per label")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Your header starts with an empty column name, which pandas usually imports as "Unnamed: 0"
    first_col = df.columns[0]
    if first_col.startswith("Unnamed") or first_col == "":
        df = df.rename(columns={first_col: "index"})

    required = ["index", "title", "content", "tags"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    if not args.no_print:
        print_contents_per_label(df, max_rows=args.max_print_rows)

    subtitles = []
    cleaned_contents = []

    for raw in df["content"].tolist():
        subtitle, cleaned = extract_subtitle_and_clean_html(raw)
        subtitles.append(subtitle)
        cleaned_contents.append(cleaned)

    out = pd.DataFrame(
        {
            "index": df["index"],
            "title": df["title"],
            "subtitle": subtitles,
            "content": cleaned_contents,
            "tags": df["tags"],
        }
    )

    out.to_csv(args.output_csv, index=False)
    print(f"\nWrote: {args.output_csv}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    main()

