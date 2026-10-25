"""Historical prompt text, extracted without executing source modules.

Original and revision prompts are separate because their instructions differ.
The inherited amazon gift card wording is retained verbatim for every product.
"""

def original_relevance(query):
    return f"""
Task: You need to determine if each of the following comments is relevant to the topic '{query}'. We are working on the amazon gift card.

Instructions:
- For each comment, state whether it is relevant or not by using the format: "[X] is relevant to the query." or "[X] is not related to the query."
- Replace '[X]' with the corresponding comment number.

Comments:
"""

def original_summary(query):
    return f"""
You are tasked with generating a high-quality summary based on user comments. Follow these steps to ensure that your summary is accurate, relevant, and well-structured.

1. Carefully Analyze the Comments:
   - Read through all the comments provided in the context.
   - Identify the key points that are related to the topic '{query}'.

2. Select Relevant Information:
   - Only include information in your summary that is relevant to the topic '{query}'.
   - For comments marked as "not relevant", simply state "[X] is not related to the query." Replace '[X]' with the corresponding comment number.

3. Construct a Coherent Summary:
   - Use an unbiased and journalistic tone in your summary.
   - Ensure that the summary is medium to long in length and that it covers the key points effectively.

4. Cite the Source of Information:
   - For each part of the summary, include a citation in the form '[NUMBER]', where 'NUMBER' corresponds to the comment's index.
   - Start numbering from '0' and continue sequentially, making sure not to skip any numbers.
   - The citation should be placed at the end of the sentence or clause that it supports.
   - If a sentence in your summary is derived from multiple comments, cite each relevant comment, e.g., '[0][1]'.

5. Final Review:
   - Double-check your citations to ensure they accurately correspond to the comments used.
   - Make sure that every sentence in the summary is cited and that irrelevant comments are correctly identified and excluded after the initial irrelevant statement.
   - Make sure every comment is cited. For example, if comment [0], [1], and [2] are all not related to the topic, then just
   summarize: '[0] is not related to the query. [1] is not related to the query. [2] is not related to the query.'
   If comment [0] is relevant, while [1], [2], and [3] are irrelevant, then summarize like this: provide a summary of [0], and then state '[1] is not related to the query. [2] is not related to the query. [3] is not related to the query.'
   Do not miss any comment even though they are irrelevant.
   - Ensure that your response is structured in JSON format with the following fields:
     - "key": A string that represents the indices of the comments used to generate this summary, e.g., "012" for comments 0, 1, and 2.
     - "summary": The final generated summary text, with citations included.

6. Key Reminders:
   - Do not include any irrelevant information in your summary. If a comment is not related to the topic, state it as described and move on.
   - Ensure that your summary is comprehensive, accurate, and clearly tied to the topic '{query}'.
"""

def original_evaluation(query):
    return f"""
    You are an AI model trained to evaluate summaries. Below, you will find several summaries identified by their labels.

    Your task is to rate each summary on one metric.

    Please make sure you read and understand every single word of these instructions.

    Evaluation Criteria:
    Information Coverage MUST be an integer from 0 to 10 - How well the summary captures and clearly describes one or several key characteristics of the
    product. A high-quality summary should convey the important features, benefits, or drawbacks of the product as highlighted in the reviews. It should
    provide a rich and accurate depiction of key points.

    Pay attention: The most important consideration is how effectively the summary communicates the product's key characteristics. The clearer and
    more richly it conveys these characteristics, the higher the score. If it fails to adequately describe the product's features, it should receive
    a low score.

    Evaluation Steps:
    1. Read all summaries provided and compare them carefully. Ensure the summary clearly and richly describes the key points relevant to the product
      without including irrelevant information.
    2. Identify any important details or characteristics of the product that are missing from the summary.
    3. Rate each of the summary based on how well it covers and conveys the important information from the reviews. The MORE comprehensively the
    summary covers the relevant information, the HIGHER the score it should receive. Pay attention: The primary focus should be on the topic
    {query}. If the summary deviates from the topic, it should receive a low score, regardless of the amount of information it contains.
    4. If a summary contains only the sentence "[X] is not related to the query." where X is a number, then give it a score of 0. However,
    if the summary contains other content besides this sentence, just ignore it when scoring.

    Your response should be in JSON format, with an array of objects. Each object should have two properties:
    1. "key": The key of the summary (e.g., "0", "1", "01", etc.)
    2. "score": The score for that summary (an integer from 0 to 10)
    """

def revision_relevance(query):
    return f"""
Task: You need to determine if each of the following comments is relevant to the topic '{query}'. We are working on the amazon gift card.

Instructions:
- For each comment, state whether it is relevant or not by using the format: "[X] is relevant to the query." or "[X] is not related to the query."
- Replace '[X]' with the corresponding comment number.

Comments:
"""

def revision_summary(query):
    return f"""
You are tasked with generating a high-quality summary based on user comments. Follow these steps to ensure that your summary is accurate, relevant, and well-structured.

1. Carefully Analyze the Comments:
   - Read through all the comments provided in the context.
   - Identify the key points that are related to the topic '{query}'.

2. Select Relevant Information:
   - Only include information in your summary that is relevant to the topic '{query}'.
   - For comments marked as "not relevant", simply state "[X] is not related to the query." Replace '[X]' with the corresponding comment number.

3. Construct a Coherent Summary:
   - Use an unbiased and journalistic tone in your summary.
   - Ensure that the summary is medium to long in length and that it covers the key points effectively.

4. Cite the Source of Information:
   - For each part of the summary, include a citation in the form '[NUMBER]', where 'NUMBER' corresponds to the comment's index.
   - Start numbering from '0' and continue sequentially, making sure not to skip any numbers.
   - The citation should be placed at the end of the sentence or clause that it supports.
   - If a sentence in your summary is derived from multiple comments, cite each relevant comment, e.g., '[0][1]'.

5. Final Review:
   - Double-check your citations to ensure they accurately correspond to the comments used.
   - Make sure that every sentence in the summary is cited and that irrelevant comments are correctly identified and excluded after the initial irrelevant statement.
   - Ensure that your response is structured in JSON format with the following fields:
     - "key": A string that represents the indices of the comments used to generate this summary, e.g., "012" for comments 0, 1, and 2.
     - "summary": The final generated summary text, with citations included.

6. Key Reminders:
   - Do not include any irrelevant information in your summary. If a comment is not related to the topic, state it as described and move on.
   - Ensure that your summary is comprehensive, accurate, and clearly tied to the topic '{query}'.
"""

def revision_evaluation(query):
    return f"""
    You are an AI model trained to evaluate summaries. Below, you will find several summaries identified by their labels.

    Your task is to rate each summary on one metric.

    Please make sure you read and understand every single word of these instructions.

    Evaluation Criteria:
    Information Coverage MUST be an integer from 0 to 10 - How well the summary captures and clearly describes one or several key characteristics of the
    product. A high-quality summary should convey the important features, benefits, or drawbacks of the product as highlighted in the reviews. It should
    provide a rich and accurate depiction of key points.

    Pay attention: The most important consideration is how effectively the summary communicates the product's key characteristics. The clearer and
    more richly it conveys these characteristics, the higher the score. If it fails to adequately describe the product's features, it should receive
    a low score.

    Evaluation Steps:
    1. Read all summaries provided and compare them carefully.
    2. Identify any important details or characteristics of the product that are missing from the summary.
    3. Rate each of the summary based on how well it covers the important information from the reviews. The MORE comprehensively the
    summary covers the relevant information, the HIGHER the score. The primary focus should be on the topic
    {query}. If the summary deviates from the topic, it should receive a low score.
    4. If a summary contains only the sentence "[X] is not related to the query." where X is a number, then give it a score of 0. However,
    if the summary contains other content besides this sentence, just ignore it when scoring.

    Your response should be in JSON format, with an array of objects. Each object should have two properties:
    1. "key": The key of the summary (e.g., "0", "1", "01", etc.)
    2. "score": The score for that summary (an integer from 0 to 10)
    """
