totalmoney = int(input('How much money do you have?'))
expense_or_income = input('Add some expense or income with description and amount:')
L = expense_or_income.split(',')
L1 = [i.split(' ') for i in L]
for l in L1:
    if (l[0] == ''):
        l.pop(0)
print(L1)
result = sum(int(l[1]) for l in L1) + totalmoney
print(result)