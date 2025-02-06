import pandas as pd
import re
from fuzzywuzzy import fuzz


# Fuzzy match function
def is_match(str1, str2, threshold=90):
    return fuzz.ratio(str1, str2) > threshold


def parse_high_low(description):
    try:
        # Splitting based on "High:" and "Low:"
        parts = description.lower().split(" high: ")
        category = parts[0]
        high_low_parts = parts[1].lower().split(" low: ")
        high_description = high_low_parts[0]
        low_description = high_low_parts[1]

        return {
            "parent_axis_name": category,
            "parent_high": high_description,
            "parent_low": low_description,
        }
    except:
        print(f"Error parsing high/low description: {description}")
        return {
            "parent_axis_name": "error",
            "parent_high": "error",
            "parent_low": "error",
        }


def parse_axis_responses(axis_response, axis_name):
    def parse_axis_responses_1(axis_response, axis_name):
        # The pattern captures the axis name, high description, low description, and the scores for models A and B
        pattern = r"- (.+?):\n\s+High: (.+?)\n\s+Low: (.+?)\n\s+Model A Score: (.+?)\n\s+Model B Score: (.+)"

        for s in axis_response.split("\n\n"):
            matches = re.match(pattern, s, re.DOTALL)
            if matches:
                paired_axis_name = matches.group(1).strip()
                if axis_name[: len(paired_axis_name)] == paired_axis_name:
                    return {
                        "scored_axis_name": matches.group(1).strip(),
                        "High": matches.group(2).strip(),
                        "Low": matches.group(3).strip(),
                        "Model A Score": matches.group(4).strip(),
                        "Model B Score": matches.group(5).strip(),
                    }
        return None

    def parse_axis_responses_2(axis_response, axis_name):
        # Adjusting the regex pattern to optionally match dashes and more flexible whitespace
        # Also adding case-insensitive matching for the axis name
        pattern = re.compile(
            r"- (\w[\w\s]*):\s*\n"  # Axis name capturing group
            r"(?:\s*-\s*)?High:\s*(.*?)\s*\n"  # High description (optional dash)
            r"(?:\s*-\s*)?Low:\s*(.*?)\s*\n"  # Low description (optional dash)
            r"(?:\s*-\s*)?Model A Score:\s*(.*?)\s*\n"  # Model A Score (optional dash)
            r"(?:\s*-\s*)?Model B Score:\s*(.*?)\s*(?=\n- |\n\n|$)",  # Model B Score (optional dash)
            re.DOTALL | re.IGNORECASE,
        )

        parsed_entries = []
        for match in re.finditer(pattern, axis_response):
            matched_axis_name = match.group(1).strip()
            # Check if the matched axis name matches the provided axis_name argument (case-insensitive)
            if matched_axis_name.lower() in axis_name.lower():
                return {
                    "scored_axis_name": matched_axis_name,
                    "High": match.group(2).strip(),
                    "Low": match.group(3).strip(),
                    "Model A Score": match.group(4).strip(),
                    "Model B Score": match.group(5).strip(),
                }
        return None

    if parse_axis_responses_1(axis_response, axis_name) is not None:
        return parse_axis_responses_1(axis_response, axis_name)
    elif parse_axis_responses_2(axis_response, axis_name) is not None:
        return parse_axis_responses_2(axis_response, axis_name)
    else:
        print(f"No matching axis found for {axis_name}")
        # raise ValueError(f"No matching axis found for {axis_name}")
        return {
            "scored_axis_name": "axis_name",
            "High": "",
            "Low": "",
            "Model A Score": "high",
            "Model B Score": "high",
        }


def extract_entities(text):
    # Regular expression to match entities: Capitalized words or phrases followed by a colon
    regex_pattern = r"-\s*(?:\*\*)?([A-Za-z ]+?)(?:\*\*)?:"
    matches = re.findall(regex_pattern, text)
    return [m for m in matches if m not in ["Model A", "Model B"]]


def extract_axis_descriptions(text):

    lines = text.strip().split("\n")

    # Initialize variables to construct the sections
    sections = []
    current_section = []

    # Process each line, building sections while excluding model scores
    for line in lines:
        # Check if the line starts a new section or is part of the current one
        if (
            line.startswith("- ") and current_section
        ):  # If starting a new section and there is a current section
            # Join the current section lines and add to sections
            sections.append(
                "\n".join(current_section).strip().replace("- ", "").replace("\n", "")
            )
            current_section = [line]  # Start a new section
        elif "Model A Score" not in line and "Model B Score" not in line:
            # If the line is not a model score, add it to the current section
            current_section.append(line)

    # Don't forget to add the last section
    if current_section:
        sections.append(
            "\n".join(current_section).strip().replace("- ", "").replace("\n", "")
        )
    return sections
