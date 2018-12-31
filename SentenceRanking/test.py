beta =0
new_coherency_score = 4
current_score = 0


numerator = (1+beta**2)*new_coherency_score*current_score
denominator = (beta**2)*new_coherency_score+current_score

print(numerator/denominator)
print(current_score*beta+new_coherency_score*(1-beta))