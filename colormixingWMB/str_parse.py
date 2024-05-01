def extract_section(response, section_name):
    """
    Extracts content sandwitched by
    --- <section_name> START ---
    --- <section_name> END ---
    """

    assert f'--- {section_name} START ---' in response and f'--- {section_name} END ---' in response, "Section markers not found in the string"
    # extract content
    content = response.split(f'--- {section_name} START ---')[1].split(f'--- {section_name} END ---')[0]
    
    return content