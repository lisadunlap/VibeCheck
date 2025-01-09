proposer_prompt_default = """
You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions. Look for differences as they pertain to content, thought process, formatting, tone, safety, presentation, and any other differences that you think could be useful for understanding the differing behavior in the two models. Remeber that not all differences are related to correctness or have a notion of good or bad and be as objective as possible.

Please format your differences as a list of axes of variation and differences between the two outputs. Try to give axes which represent a property that a human could easily interpret and they could categorize a pair of text outputs as higher or lower on that specific axis.

The format should be
- {{axis_1}}: {{difference}}
- {{axis_2}}: {{difference}}
    
If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

# proposer_prompt_freeform = """
# You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions. Write down as many differences as you can find between the two outputs. Please format your differences as a list of axes of variation and differences between the two outputs. Try to give axes which represent a property that a human could easily interpret and they could categorize a pair of text outputs as higher or lower on that specific axis. If the axis is related to a specific kind of question (e.g. math, coding, creative writing), please specify that in the axis name.

# The format should be
# - {{axis_1}}: {{difference}}
# - {{axis_2}}: {{difference}}

# If there are no substantive differences between the outputs, please respond with only "No differences found."
# """

proposer_prompt_freeform = """
You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions. Write down as many differences as you can find between the two outputs. Please format your differences as a list of axes of variation and differences between the two outputs. Try to give axes which represent a property that a human could easily interpret and they could categorize a pair of text outputs as higher or lower on that specific axis. 

The format should be
- {{axis_1}}: {{difference}}
- {{axis_2}}: {{difference}}
    
If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

proposer_prompt_freeform_no_detail = """
You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions. Write down as many differences as you can find between the two outputs. Please format your differences as a list of axes of variation and differences between the two outputs. Try to give axes which represent a property that a human could easily interpret and they could categorize a pair of text outputs as higher or lower on that specific axis. Do not list differences that are about the level of detail in the response, but rather differences in the overall content, style, or approach.

The format should be
- {{axis_1}}: {{difference}}
- {{axis_2}}: {{difference}}
    
If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

proposer_prompt_freeform_summarization = """
You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions for a summarization task. What do these different models focus on when summarizing? How do their styles differ? Write down as many differences as you can find between the two outputs. Please format your differences as a list of axes of variation and differences between the two outputs. Try to give axes which represent a property that a human could easily interpret and they could categorize a pair of text outputs as higher or lower on that specific axis. If the axis is related to a specific kind of question (e.g. math, coding, creative writing), please specify that in the axis name.

The format should be
- {{axis_1}}: {{difference}}
- {{axis_2}}: {{difference}}
    
If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

proposer_prompt_freeform_summarization_objective = """
You are a machine learning researcher trying to analyze the differences between the behaviors of two llms by finding objective differences in their responses to the same set of questions for a summarization task. Focus on observable, measurable differences in what content they include, their structure, and their linguistic patterns. Write down as many objective differences as you can find between the two outputs. Please format your differences as a list of axes of variation and differences between the two outputs. Each axis should represent a concrete, measurable property that can be identified consistently across different summaries. For example: length, use of direct quotes, inclusion of specific types of information, sentence structure patterns, etc.

The format should be
- {{axis_1}}: {{difference}}
- {{axis_2}}: {{difference}}
    
If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

proposer_prompt_freeform_vlm = """
You are a machine learning researcher trying to figure out the major differences between the behaviors of two VLM by finding differences in their responses to the same set of VQA questions (the images have been removed). Write down as many differences as you can find between the two outputs. Please format your differences as a list of axes of variation and differences between the two outputs. Try to give axes which represent a property that a human could easily interpret and they could categorize a pair of text outputs as higher or lower on that specific axis. If the axis is related to a specific kind of question (e.g. math, coding, creative writing), please specify that in the axis name.

The format should be
- {{axis_1}}: {{difference}}
- {{axis_2}}: {{difference}}
    
If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

proposer_prompt_creative_writing = """
You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of creative writing prompts. What is the difference in writing style, format, content, etc? Write down as many differences as you can find between the two outputs and be specific. Please format your differences as a list of axes of variation and differences between the two outputs. Try to give axes which represent a property that a human could easily interpret and they could categorize a pair of text outputs as higher or lower on that specific axis. If the axis is related to a specific kind of question (e.g. math, coding, creative writing), please specify that in the axis name.

The format should be
- {{axis_1}}: {{difference}}
- {{axis_2}}: {{difference}}

If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

proposer_prompt_creative_stem = """
You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of STEM and coding questions. Write down as many differences as you can find between the two outputs. Please format your differences as a list of axes of variation and differences between the two outputs. Try to give axes which represent a property that a human could easily interpret and they could categorize a pair of text outputs as higher or lower on that specific axis. If the axis is related to a specific kind of question (e.g. math, coding, biology), please specify that in the axis name.

The format should be
- {{axis_1}}: {{difference}}
- {{axis_2}}: {{difference}}

If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

proposer_prompt_math_cot = """
You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions. Look for differences as they pertain to math content, clarity, correctness, and other qualitative differences that are important in math as well as any other axes like prompt adherence, tone, formatting, or anything else that you think is important.

Please output a list of qualitative differences between these sets of outputs with relation to specific axes of variation. Try to give axes which represent a qualiative property that a human could easily interpret and they could understand what it means to be higher or lower on that specific axis. Please ensure that the concepts used to explain what is high and low on the axis are distinct and mutually exclusive such that given any tuple of text outputs, a human could easily and reliably determine which model is higher or lower on that axis. 

The format should be
- {{axis_1}}: {{difference}}
- {{axis_2}}: {{difference}}
    
    Please output differences which have a possibility of showing up in future unseen data and which would be useful for a human to know about when deciding with LLM to use. Please do not list any axes which relate to accuracy, as the goal is to find qualitative differences which may correlate with human preference. For each axis, define clearly and succinctly what constitutes a high or low score, ensuring these definitions are mutually exclusive. For each axis, also provide an explanation of what in the text proves this difference exists. Please order your response in terms of the most prominent differences between the two outputs. If there are no substantive differences between the outputs, please respond with only "No differences found."
"""
