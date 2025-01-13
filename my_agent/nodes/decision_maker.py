from state import State

def decide_to_go_next(state: State):
    """
    Decide whether to go to the next node based on the current state

    Args:
        state (State): Current state of the conversation
    Returns:
        str: Name of the next node to be executed
    """

    if state['web_search'] == 'yes':
        return 'transform_query'
    
    return 'generator'


def decide_response_generator(state: State):
    """
    Decide which response generator to use based on the current state

    Args:
        state (State): Current state of the conversation
    Returns:
        str: Name of the response generator to be used
    """

    category = state['category']

    if category == 'greeting' or category == 'irrelevant':
        return 'llm'
    
    if category == 'relevant':
        return 'retriever'
    
    return 'llm'