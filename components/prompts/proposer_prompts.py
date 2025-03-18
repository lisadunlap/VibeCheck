proposer_freeform = """You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions and seeing if these differences correspond with user preferences. Write down as many differences as you can find between the two outputs. Please format your differences as a list of properties that appear more in one output than the other.

Below are multiple sets of questions and responses, separated by dashed lines. For each set, analyze the differences between Model 1 and Model 2. What properties are seen in the responses from Model 1 that are not seen in the responses from Model 2? What properties are seen in the responses from Model 2 that are not seen in the responses from Model 1?

{combined_responses}

The format should be a list of properties that appear more in one output than the other in the format of a short description of the property. 

Consider differences on many different axes such as tone, language, structure, content, safety, and any other axis that you can think of. If the questions have a specific property or cover a specific topic, also consider differences which are relevant to that property or topic.
    
Remember that these differences should be human interpretable and that the differences should be concise, substantive and objective. Write down as many properties as you can find. Do not explain which model has which property, simply describe the property. Your response should not include any mention of Model 1 or Model 2.

Respond with a list of properties, each on a new line separated by *. Do NOT include any other text in your response. If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

proposer_freeform_iteration = """You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions and seeing if these differences correspond with user preferences. I have already found some differences between the two outputs, but there are many more differences to find. Write down as many differences as you can find between the two outputs which are not already in the list of differences. Please format your differences as a list of properties that appear more in one output than the other.

Below are multiple sets of questions and responses, separated by dashed lines. For each set, analyze the differences between Model 1 and Model 2. What properties are seen in the responses from Model 1 that are not seen in the responses from Model 2? What properties are seen in the responses from Model 2 that are not seen in the responses from Model 1? Here are the differences I have already found and the questions and responses:

{combined_responses}

The format should be a list of properties that appear more in one output than the other in the format of a short description of the property.

Consider differences on many different axes such as tone, language, structure, content, safety, and any other axis that you can think of. If the questions have a specific property or cover a specific topic, also consider differences which are relevant to that property or topic.
    
Remember that these differences should be human interpretable and that the differences should be concise, substantive and objective. Write down as many properties as you can find which are not already represented in the list of differences. Do not explain which model has which property, simply describe the property. Your response should not include any mention of Model 1 or Model 2.
Respond with a list of new properties, each on a new line separated by *. Do NOT include any other text in your response. If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

proposer_onesided = """You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions and seeing if these differences correspond with user preferences. Write down as many properties as you can find that are present in Model 1 but not in Model 2. Please format your differences as a list of properties that appear more in one output than the other.

{combined_responses}

The format should be a list of properties that appear more in the output of Model 1 than the output of Model 2 in the format of a short description of the property. Respond with a list of properties, each on a new line.

Consider differences on many different axes such as tone, language, structure, content, safety, and any other axis that you can think of. If the questions have a specific property or cover a specific topic, also consider differences which are relevant to that property or topic.
    
Remember that these differences should be human interpretable and that the differences should be concise, substantive and objective. Write down as many properties as you can find. Do not explain which model has which property, simply describe the property. Your response should not include any mention of Model 1 or Model 2, only the properties that are present more in Model 1 than Model 2.
If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

proposer_onesided_iteration = """You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions and seeing if these differences correspond with user preferences. I have already found some properties, but there are many more properties to find. Write down as many properties as you can find that are present in Model 1 but not in Model 2.

Below are multiple sets of questions and responses, separated by dashed lines. For each set, analyze the differences between Model 1 and Model 2. What properties are seen in the responses from Model 1 that are not seen in the responses from Model 2? What properties are seen in the responses from Model 2 that are not seen in the responses from Model 1? Here are the differences I have already found and the questions and responses:

{combined_responses}

The format should be a list of properties that appear more in the output of Model 1 than the output of Model 2 in the format of a short description of the property. Respond with a list of properties, each on a new line.

Note that this example is not at all exhaustive, but rather just an example of the format. Consider differences on many different axes such as tone, language, structure, content, safety, and any other axis that you can think of. If the questions have a specific property or cover a specific topic, also consider differences which are relevant to that property or topic.
    
Remember that these differences should be human interpretable and that the differences should be concise, substantive and objective. Write down as many properties as you can find which are not already represented in the list of differences. Do not explain which model has which property, simply describe the property. Your response should not include any mention of Model 1 or Model 2, only the properties that are present more in Model 1 than Model 2. If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

proposer_freeform_axis = """You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions. Write down as many differences as you can find between the two outputs. Please format your differences as a list of axes of variation and differences between the two outputs. Try to give axes which represent a property that a human could easily interpret and they could categorize a pair of text outputs as higher or lower on that specific axis. 

Here are the questions and responses:
{combined_responses}

The format should be a list of axes in the format of {{axis}}: High: {{high description}} Low: {{low description}} for each axis, with each axis on a new line separated by *. Do NOT include any other text in your response.

Consider differences on many different axes such as tone, language, structure, content, safety, and any other axis that you can think of. If the questions have a specific property or cover a specific topic, also consider differences which are relevant to that property or topic.
If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

proposer_freeform_iteration_axis = """You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions and seeing if these differences correspond with user preferences. I have already found some differences between the two outputs, but there are many more differences to find. Write down as many differences as you can find between the two outputs which are not already in the list of differences. Please format your differences as a list of properties that appear more in one output than the other.

Below are multiple sets of questions and responses, separated by dashed lines. For each set, analyze the differences between Model 1 and Model 2. Please format your differences as a list of axes of variation and differences between the two outputs. Try to give axes which represent a property that a human could easily interpret and they could categorize a pair of text outputs as higher or lower on that specific axis. 

Here are the differences I have already found and the questions and responses:

{combined_responses}

The format should be a list of axes in the format of {{axis}}: High: {{high description}} Low: {{low description}} for each axis, with each axis on a new line separated by *. Do NOT include any other text in your response.

Consider differences on many different axes such as tone, language, structure, content, safety, and any other axis that you can think of. If the questions have a specific property or cover a specific topic, also consider differences which are relevant to that property or topic.
    
Remember that these differences should be human interpretable and that the differences should be concise, substantive and objective. Write down as many properties as you can find which are not already represented in the list of differences. Do not explain which model has which property, simply describe the property. Your response should not include any mention of Model 1 or Model 2.
Respond with a list of new properties, each on a new line separated by *. Do NOT include any other text in your response. If there are no substantive differences between the outputs, please respond with only "No differences found."
"""