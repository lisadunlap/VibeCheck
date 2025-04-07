reduce_freeform = """Below is a list of properties that are found in LLM outputs. I would like to summarize this list to a set of representative properties with clear and concise descriptions that cover the recurring themes in the data. Are there any interesting overarching properties that are present in a large number of the properties? Please return a list of properties that are seen in the data, where each property represents one type of behavior that is seen in the data.

Here is the list of properties:
{differences}

A human should be able to understand the property and its meaning, and this property should provide insight into the model's behavior or personality. Do not include subjective analysis about these properties, simply describe the property. For instance "the model is more advanced in its understanding" and "the model uses historical context" is not a good property because it is too vague and does not provide interesting insight into the model's behavior. Similarly, these properties should be on a per prompt basis, so "the model provides a consistent tone across prompts" or "the model varies its tone from formal to informal" is not a good property because a person could not make a judgement only looking at a single prompt.

These properties should be something that a human could reasonably expect to see in the model's output when given new prompts. If the property is specific to a type of task (e.g. coding), please ensure that the axis is named in a way that makes it clear what type of task it applies to in the axis name.

Order your final list of properties by how much they are seen in the data. Your response should be a list deliniated with "-"
"""

proposer_fixed_difference = """Below is a list of properties that are found in LLM outputs. I would like to summarize this list to AT MOST {num_reduced_axes} representative properties with clear and concise descriptions. Are there any interesting overarching properties that are present in a large number of the properties?

Here is the list of properties:
{differences}

A human should be able to understand the property and its meaning, and this property should provide insight into the model's behavior or personality. Do not include subjective analysis about these properties, simply describe the property. For instance "the model is more advanced in its understanding" or "the model uses historical context" is not a good property because it is too vague and does not provide interesting insight into the model's behavior. These properties should be something that a human could reasonably expect to see in the model's output when given new prompts. Order your final list of properties by how much they are seen in the data. Your response should be a list deliniated with "-"
"""

reduce_freeform_axis = """The following are the axes of variation that you can consider when comparing the two model outputs along with a description of how two models (A and B) vary along that axis. Each axis has a name as well as a description of what it means to be low and high on this axis. Many of these axes of variations could be named incorrectly or redundant with other axes. I want to reduce this list of axes to a minimal set of parent axes that cover the majority of the axes. Please ensure these parent axes' descriptions of what makes an item high or low on that axis align with the high and low descriptions of the axes they cover. Your new set of axes should be distinct such that each of the above axes fit under at most one of your new axes. Please ensure that the descriptions of what makes a piece of text low or high on this axis are distinct and mutually exclusive such that given any pair of text outputs, a human could easily and reliably determine which model is higher or lower on that axis. 

If an axis applies to a specific type of task (e.g. coding), please ensure that the axis is named in a way that makes it clear what type of task it applies to in the axis name.
                        
Here are the axes of varaiation (note each axis is formatted {{axis name}}: High: {{high description}} Low: {{low description}}):
{differences}

Again I want to cluster these axes into a minimal set of parent axes that cover the entire axis list. Please ensure these parent axes' descriptions of what makes an item high or low on that axis align with the high and low descriptions of the axes they cover. Your new set of axes should be distinct so each of the above axes fit under exactly one of your new axes. Please ensure each axis and parent axis contains an axis name and descriptions of what it means to score high or low on that axis in the same format as the provided axes. If the property is specific to a type of task (e.g. coding), please ensure that the axis is named in a way that makes it clear what type of task it applies to in the axis name. Please ensure the descriptions of what is considered high and low on each axis is clear, concise, under 20 words.

Your response should be a list of axes in the format of {{axis name}}: High: {{high description}} Low: {{low description}} for each axis. Order your final list of axes by how much they are seen in the data. Your response should be a list deliniated with "-"
"""

reduce_freeform_axis_longer = """The following are the axes of variation that you can consider when comparing the two model outputs along with a description of how two models (A and B) vary along that axis. Each axis has a name as well as a description of what it means to be low and high on this axis. Many of these axes of variations could be named incorrectly or redundant with other axes. I want to reduce this list of axes to a minimal set of parent axes that cover the majority of the axes. Please ensure these parent axes' descriptions of what makes an item high or low on that axis align with the high and low descriptions of the axes they cover. Your new set of axes should be distinct such that each of the above axes fit under at most one of your new axes. Please ensure that the descriptions of what makes a piece of text low or high on this axis are distinct and mutually exclusive such that given any pair of text outputs, a human could easily and reliably determine which model is higher or lower on that axis. 

If an axis applies to a specific type of task (e.g. coding), please ensure that the axis is named in a way that makes it clear what type of task it applies to in the axis name.
                        
Here are the axes of varaiation (note each axis is formatted {{axis name}}: High: {{high description}} Low: {{low description}}):
{differences}

Again I want to cluster these axes into a minimal set of parent axes that cover the entire axis list. Please ensure these parent axes' descriptions of what makes an item high or low on that axis align with the high and low descriptions of the axes they cover. Your new set of axes should be distinct so each of the above axes fit under exactly one of your new axes. Please ensure each axis and parent axis contains an axis name and descriptions of what it means to score high or low on that axis in the same format as the provided axes. If the property is specific to a type of task (e.g. coding), please ensure that the axis is named in a way that makes it clear what type of task it applies to in the axis name.

Your response should be a list of axes in the format of {{axis name}}: High: {{high description}} Low: {{low description}} for each axis. Order your final list of axes by how much they are seen in the data. Your response should be a list deliniated with "-"
"""