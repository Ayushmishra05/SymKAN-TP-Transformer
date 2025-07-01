def preprocess_expression(expr):
    expr = expr.replace(' * ', '*').replace(' / ', '/').replace(' ^ ', '^')
    expr = expr.replace(' + ', '+').replace(' - ', '-')
    expr = expr.replace("+-" , "-") 
    expr = expr.replace("-+" , "-") 
    expr = ' '.join(expr.split())
    expr = expr.replace('me', 'm_e')  # Example rule, adjust based on dataset
    return expr 



amp = r"-1/2*i*e^2*gamma_{+\INDEX_0,INDEX_1,INDEX_2}*gamma_{\INDEX_0,INDEX_3,INDEX_4}*e_{MOMENTUM_0,INDEX_2}(p_1)_u*e_{MOMENTUM_1,INDEX_4}(p_2)_u*e_{MOMENTUM_2,INDEX_1}(p_3)_u^(*)*e_{MOMENTUM_3,INDEX_3}(p_4)_u^(*)/(m_e^2+-s_13+1/2*reg_prop)"
sqamp = r"2*e^4*(m_e^4 + -1/2*m_e^2*s_13 + 1/2*s_14*s_23 + -1/2*m_e^2*s_24 + 1/2*s_12*s_34)*(m_e^2 + -s_13 + 1/2*reg_prop)^(-2)"

amp = preprocess_expression(amp)
sqamp = preprocess_expression(sqamp)
print(amp , "\n\n\n\n")
print(sqamp)