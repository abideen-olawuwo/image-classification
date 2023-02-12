def compound_interest(principal, rate, time):
    #Calculate compound interest
    Amount = principal * (pow((1 + rate / 100), time))
    CI = Amount - principal
    print("Compound interest is", CI)

P = float(input("Enter Principal Value : "))
R = float(input("Enter Rate Value : "))
T = float(input("Enter Time Value : "))
compound_interest(P,R,T)
