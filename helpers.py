import re

def resolve_ae(text):
    """
    This function takes a text input in Central Kurdish (Sorani) script and performs a series of character replacements
    to standardize variations in the script. Specifically, it addresses cases where the character 'ە' (Arabic letter
    AE) may be used in different contexts.
    """
    # First replace all occurrences of 'ه' with 'ە'
    text = re.sub('ه', 'ە', text)
    # Replace specific combinations with 'ها', 'هێ', and 'ه'
    text = re.sub("ەا", "ها", text) # Replace ەا with ها
    text = re.sub("ەێ", "هێ", text) # Replace ەێ with هێ
    # Replace ە (AE) at the beginning of a word with ه (HEH)
    text = re.sub(r'\b(ە\w*)', lambda match: 'ه' + match.group(1)[1:], text)
    #  Replace ALEF+AE with ALEF+HEH
    text = re.sub("اە", "اه", text)
    # Replace 'ەە' at the beginning and end with 'هە'
    text = re.sub(r'\bەە|ەە\b', 'هە', text)
    # Replace 'ەە'AE+AE in the middle of a word with HEH+AE
    text = re.sub(r'ەە(?=\w)', 'ەه', text)
    # Replace two AE with spaces in between with AE HEH
    text = re.sub("ە ە", "ە ه", text)
    return text


if __name__ == "__main__":

    # test case  ەا with ها
    AE_AND_ALEF = "هەروەەا جۆرەەا"
    assert resolve_ae(AE_AND_ALEF) == "هەروەها جۆرەها"

    # test case ەێ with هێ
    AE_AND_YEH_WITH_SMALL_V = "ەێمن ەێژا ەێڤی"
    assert resolve_ae(AE_AND_YEH_WITH_SMALL_V) == "هێمن هێژا هێڤی"
    
    AE_BEGINNING = "ەێڵ ەەڵوێست ەەژار ەەڵە"
    assert resolve_ae(AE_BEGINNING) == "هێڵ هەڵوێست هەژار هەڵە"

    ALEF_AE = "ئاەەنگ ئاەورا"
    assert resolve_ae(ALEF_AE) == "ئاهەنگ ئاهورا"

    TWO_AE_BEGINNING = "ەەوار ەەوراز"
    assert resolve_ae(TWO_AE_BEGINNING) == "هەوار هەوراز"

    TWO_AE_END = "دەەە وەەە"
    assert resolve_ae(TWO_AE_END) == "دەهە وەهە"

    TWO_AE_MIDDLE = "نەەەنگ"
    assert resolve_ae(TWO_AE_MIDDLE) == "نەهەنگ"


    text = "ەەروەەا جۆرهەا ەێمن ەڵوێست ەهڵە ەهوراز ەەوراز "
    print(resolve_ae(text))