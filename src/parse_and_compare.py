import sympy as sp
from typing import List, Optional
from preprocess.tokenizersplit import reconstruct_expression

def parse_and_compare(ground_truth_str: str, obtained_tokens: List[int], vocab: 'SymbolicVocab', 
                     tokenizer: 'SymbolicQEDTokenizer') -> bool:
    """
    Compare ground truth squared amplitude with model's predicted tokens for symbolic equivalence.
    
    Args:
        ground_truth_str: Ground truth squared amplitude (e.g., "16*m_d^2*m_u^2").
        obtained_tokens: Model output as token indices (e.g., [23, 5, 27, 11, 10, ...]).
        vocab: SymbolicVocab instance for decoding token indices.
        tokenizer: SymbolicQEDTokenizer instance for reconstructing expressions.
    
    Returns:
        bool: True if expressions are symbolically equivalent, False otherwise.
    """
    try:
        # Decode token indices to tokens
        decoded_tokens = vocab.decode(obtained_tokens, include_special_tokens=False)
        # Reconstruct expression from tokens
        obtained_str = reconstruct_expression(decoded_tokens)
        
        # Replace ^ with ** for SymPy parsing
        ground_truth_str = ground_truth_str.replace('^', '**')
        obtained_str = obtained_str.replace('^', '**')
        
        # Define SymPy symbols for all QED terms
        i, e = sp.symbols('i e')
        # Include all mass terms as per dataset
        m_b, m_c, m_u, m_d, m_s, m_e, m_t, m_mu = sp.symbols('m_b m_c m_u m_d m_s m_e m_t m_mu')
        s_12, s_13, s_14, s_23, s_24, s_34 = sp.symbols('s_12 s_13 s_14 s_23 s_24 s_34')
        reg_prop = sp.symbols('reg_prop')
        
        locals_dict = {
            'i': i, 'e': e,
            'm_b': m_b, 'm_c': m_c, 'm_u': m_u, 'm_d': m_d, 'm_s': m_s, 'm_e': m_e, 'm_t': m_t, 'm_mu': m_mu,
            's_12': s_12, 's_13': s_13, 's_14': s_14, 's_23': s_23, 's_24': s_24, 's_34': s_34,
            'reg_prop': reg_prop
        }
        
        # Parse expressions
        ground_truth_expr = sp.sympify(ground_truth_str, locals=locals_dict)
        obtained_expr = sp.sympify(obtained_str, locals=locals_dict)
        
        # Check symbolic equivalence
        diff = sp.simplify(ground_truth_expr - obtained_expr)
        equivalent = diff == 0
        
        if not equivalent:
            print(f"Ground Truth: {ground_truth_str}")
            print(f"Predicted: {obtained_str}")
            print(f"Difference (simplified): {diff}")
        
        return equivalent
    
    except sp.SympifyError as e:
        print(f"SymPy parsing error: {e}")
        return False
    except Exception as e:
        print(f"Error in parse_and_compare: {e}")
        return False