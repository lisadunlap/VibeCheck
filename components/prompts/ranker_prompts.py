ranker_prompt = """You are a fair and unbiased judge. Your task is to compare the outputs of two language models (A and B) on a given property. Which response better aligns more with the given property, A, B, or equal?

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

Here are the properties and the two responses:
{inputs}

Remember to be as objective as possible and strictly adhere to the response format."""

ranker_prompt_multi = """You are a fair and unbiased judge. Your task is to compare the outputs of two language models (A and B) on a set of one or more properties. Which response better aligns more with each property, A, B, or equal?

Your sole focus is to determine which response better aligns with each property, NOT how good or bad the response is. Consider each property independently. Do NOT let the position of the model outputs influence your decision and remain as objective as possible. Consider what each property means and how it applies to the outputs. Would a reasonable person be able to tell which output aligns more with the property based on the description?

Instructions: For each property,
	•	If Response A aligns with the property more than Response B, respond with "A".
  •	If Response B aligns with the property more than Response A, respond with "B".
	•	If the responses are roughly equal on the property or neither response contains the property, respond with "equal". 
	•	If the property does not apply to these outputs (e.g., the property is about code quality, but the prompt is not related to coding), respond with "N/A".
	•	If you are unsure about the meaning of the property, respond with "unsure". Think about of a reasonable person would find the property easy to understand.

A group of humans should agree with your decision. The properties will be given in the form of a numbered list. Use the following format for your response:
Ranking:
Property 1: {{A, B, equal, N/A, or unsure}}
Analysis: {{reasoning}}
Property 2: {{A, B, equal, N/A, or unsure}}
Analysis: {{reasoning}}
Property 3: {{A, B, equal, N/A, or unsure}}
Analysis: {{reasoning}}
...

Here are the properties and the two responses:
{inputs}

Remember to be as objective as possible and strictly adhere to the response format. You must give a ranking for each property."""

ranker_prompt_axis = """You are a fair and unbiased judge. Your task is to compare the outputs of two language models (A and B) on a given axis. Each axis contains a description explaining what it means for an output to be high or low. Your goal is to decide which model's output is higher on the axis.
When comparing the outputs, consider the following:

	•	Being high or low on the axis does not indicate how good or bad the response is. Your sole focus is to order the responses based on how they differ on this axis.
	•	Avoid any position bias and remain as objective as possible.

Instructions:
	•	If Response A aligns with the 'high' description more than Response B, respond with "A".
	•	If Response B aligns with the 'high' description more than Response A, respond with "B".
	•	If the responses are roughly equal on the axis, respond with "equal". Use this sparingly, most of the time you should respond with "A" or "B".
	•	If the axis does not apply to these outputs (e.g., the axis is about code quality, but the prompt is not related to coding), respond with "N/A".
	•	If you are unsure about the meaning of the axis, respond with "unsure".

Use the following format for your response:

Analysis: {{reasoning}}
Model: {{A, B, equal, N/A, or unsure}}

Here are the properties and the two responses:
{inputs}

Remember to be as objective as possible and strictly adhere to the response format."""

ranker_prompt_axis_multi = """You are a fair and unbiased judge. Your task is to compare the outputs of two language models (A and B) on a set of one or more properties. Each property contains a description explaining what it means for an output to be high or low on that property. Your goal is to decide which model's output is higher on each property.

Instructions: For each property,
	•	If Response A aligns with the 'high' description more than Response B, respond with "A".
  •	If Response B aligns with the 'high' description more than Response A, respond with "B".
	•	If the responses are roughly equal on the property or neither response contains the property, respond with "equal". Use this sparingly, most of the time you should respond with "A" or "B".
	•	If the property does not apply to these outputs (e.g., the property is about code quality, but the prompt is not related to coding), respond with "N/A".
	•	If you are unsure about the meaning of the property, respond with "unsure".

A group of humans should agree with your decision. The properties will be given in the form of a numbered list. Use the following format for your response:
Ranking:
Property 1: {{A, B, equal, N/A, or unsure}}
Analysis: {{reasoning}}
Property 2: {{A, B, equal, N/A, or unsure}}
Analysis: {{reasoning}}
Property 3: {{A, B, equal, N/A, or unsure}}
Analysis: {{reasoning}}
...

Here are the properties and the two responses:
{inputs}

Remember to be as objective as possible and strictly adhere to the response format. You must give a ranking for each property. Do not give any other information in your response."""


judge_prompt = """You are an impartial judge and evaluate the quality of the responses provided by two AI assistants (A and B) to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 

Here is the prompt and the outputs of A and B respectively:

{inputs}

Please respond with the model which contains a higher quality response. Based on your analysis, please explain your reasoning before assigning a score. Use the following format for your response:
Analysis: {{reasoning}}
Model: {{A, B, tie}}
"""