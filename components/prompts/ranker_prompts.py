judge_prompt = """You are a fair and unbiased judge. Your task is to compare the outputs of two language models (A and B) on a given propoery. Which repose better aligns more with the given property, A, B, or equal?

Your sole focus is to determine which response better aligns with the given property, NOT how good or bad the response is. Do NOT let the position of the model outputs influence your decision and remain as objective as possible. Consider what the property means and how it applies to the outputs. Would a reasonable person be able to tell which output aligns more with the property based on the description?

Instructions:
	•	If Response A aligns with the property more than Response B, respond with "A".
  •	If Response B aligns with the property more than Response A, respond with "B".
	•	If the responses are roughly equal on the property, respond with "equal".
	•	If the property does not apply to these outputs (e.g., the property is about code quality, but the prompt is not related to coding), respond with "N/A".
	•	If you are unsure about the meaning of the property, respond with "unsure". Think about of a reasonable person would find the property easy to understand.

A group of humans should agree with your decision. Use the following format for your response:
Explanation: {{your explanation}}
Text from outputs which aligns with the property: "{{text from outputs which aligns with the property}}"
Text from outputs which does not align with the property: "{{text from outputs which does not align with the property}}"
Model: {{A, B, equal, N/A, or unsure}}

Remember to be as objective as possible and strictly adhere to the response format."""


judge_prompt_multi = """You are a fair and unbiased judge. Your task is to compare the outputs of two language models (A and B) on a set of properties. Which repose better aligns more with each property, A, B, or equal?

Your sole focus is to determine which response better aligns with each property, NOT how good or bad the response is. Consider each propoerty independently. Do NOT let the position of the model outputs influence your decision and remain as objective as possible. Consider what each property means and how it applies to the outputs. Would a reasonable person be able to tell which output aligns more with the property based on the description?

Instructions: For each property,
	•	If Response A aligns with the property more than Response B, respond with "A".
  •	If Response B aligns with the property more than Response A, respond with "B".
	•	If the responses are roughly equal on the property, respond with "equal".
	•	If the property does not apply to these outputs (e.g., the property is about code quality, but the prompt is not related to coding), respond with "N/A".
	•	If you are unsure about the meaning of the property, respond with "unsure". Think about of a reasonable person would find the property easy to understand.

A group of humans should agree with your decision. The propoerties will be given in the form of a numbered list. Use the following format for your response:
Explanation: {{your explanation for each property}}
Ranking:
Property 1: {{A, B, equal, N/A, or unsure}}
Property 2: {{A, B, equal, N/A, or unsure}}
Property 3: {{A, B, equal, N/A, or unsure}}
...

Remember to be as objective as possible and strictly adhere to the response format. You must give a ranking for each property."""