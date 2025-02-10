LLM_ONLY_PROMPT = """
    The following are the result of asking two different language models to generate an answer for the same questions:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can correctly identify which model generated which response for unseen questions. This is a very samll portion of the data, so I want the differences to be general.

    Please output a list (separated by bullet points "*") of distinct concepts or styles that appear more in the outputs of Model A compared to Model B. An example of the desired output format:
    * "casual language"
    * "lists with repeating phrases"
    * "polite language"

    Do not have "model A" or "model B" in your response. Please order your response in terms of the most common differences between the two models. Your response:
"""

LLM_ONLY_PROMPT_DUAL = """
    The following are the result of asking two different language models to generate an answer for the same questions:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can correctly identify which model generated which response for unseen questions. This is a very small portion of the data, so I want the differences to be general.

    Please output a list dictionary of distinct concepts or styles that appear more in the outputs of Model A compared to Model B and vice versa. An example of the desired output format:

   {{"Model A contains more": ["casual language", "lists with repeating phrases", "polite language"], "Model B contains more": ["formal language", "short responses"]}}

   If there are no differences, write an expty list. For example, {{"Model A contains more": [], "Model B contains more": []}} 

    Each difference should be less than 10 words. Your response:
"""


LLM_ONLY_PROMPT_DUAL2 = """
    The following are the result of asking two different language models to generate an answer for the same questions:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can better compare the behavior of these models in a qualitative way.

    Please output a list dictionary of distinct concepts or styles that appear more in the outputs of Model A compared to Model B and vice versa. An example of the desired output format:

   {{"Model A contains more": ["lack of reponse", ...], "Model B contains more": ["repeating phrases", ...]}}

   If there are no differences, write an expty list. For example, {{"Model A contains more": [], "Model B contains more": []}} 

    Each difference should be less than 10 words. Your response:
"""

LLM_ONLY_PROMPT_OZ = """
    The following are the result of asking two different language models to generate an answer for the same questions:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two LLM outputs so I can better compare the behavior of these models in a qualitative way.

    Please output a list dictionary of distinct concepts or styles that appear more in the outputs of Model A compared to Model B and vice versa. An example of the desired output format:

   {{"Model A contains more": ["lack of reponse", ...], "Model B contains more": ["repeating phrases", ...]}}

   If there are no differences, write an expty list. For example, {{"Model A contains more": [], "Model B contains more": []}} 

   Here are a list of axes of variation to consider:

   {axes}

   This list is not exhaustive, so please add any differences even if they do not relate to the above axes. If the outputs are roughly the same along one of the provided axes do not include it. Please output differences which can be understood by a human without having to read through the entire text.

    Each difference should be less than 10 words. Please order your response in terms of the most prominent differences between the two outputs.
"""

LLM_ONLY_PROMPT_WRITING = """
    The following are the result of asking two different language models to generate an answer for the same writing questions:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can correctly identify which model generated which response for unseen questions. This is a very samll portion of the data, so I want the differences to be general.

    Please output a list (separated by bullet points "*") of distinct concepts or styles that appear more in the outputs of Model A compared to Model B. An example of the desired output format:
    * "casual language"
    * "lists with repeating phrases"
    * "polite language"

    Do not have "model A" or "model B" in your response. Please order your response in terms of the most common differences between the two models and output a list (separated by bullet points "*"). Your response:
"""
