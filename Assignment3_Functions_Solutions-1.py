import numpy as np

#====================================================================================================
# SIR Model differential equations Helper Function
#====================================================================================================
def sir_model(S_t, I_t, R_t, beta, gamma):
    """'
    This function takes in a specified infection rate (beta), recovery rate (gamma) and the
    current number of susceptible (S), infected (I) and recovered individuals (R) for a given day.
    It returns the rate of change in the number of susceptible individuals (dS/dt), change in the number
    of infected individuals (dI/dt), and change in the number of recovered individuals (dR/dt) for that day.
    """
    dS_dt = -beta * ((S_t*I_t)/(S_t+I_t+R_t)) #Change in number of susceptible individuals at day t
    dI_dt  = beta * ((S_t*I_t)/(S_t+I_t+R_t)) - gamma*I_t #Change in number of infected people at day t
    dR_dt = gamma*I_t #Change in number of recovered individuals at day t
    return dS_dt, dI_dt, dR_dt

#Note: Instead of using a fixed N, use S+I+R because N=S+I+R. (When doing the sird_model, you have to do this
# because N can change since people die. So it's best not to feed in a fixed N and use S+I+R).


#====================================================================================================
# SIRD Model differential equations Helper Function
#====================================================================================================
def sird_model(S_t, I_t, R_t, beta, gamma, mu):
    """'
    This function takes in a specified infection rate (beta), recovery rate (gamma), death rate (mu) and the
    current number of susceptible (S), infected (I) and recovered individuals (R) for a given day.
    It returns the rate of change in the number of susceptible individuals (dS/dt), change in the number
    of infected individuals (dI/dt),  change in the number of recovered individuals (dR/dt),
    and change in the number of deaths (dD/dt) for that day.
    """
    dS_dt = -beta * ((S_t*I_t)/(S_t+I_t+R_t)) #Change in number of susceptible individuals at day t
    dI_dt  = beta * ((S_t*I_t)/(S_t+I_t+R_t)) - gamma*I_t - mu*I_t #Change in number of infected people at day t
    dR_dt = gamma*I_t #Change in number of recovered individuals at day t
    dD_dt = mu*I_t
    return dS_dt, dI_dt, dR_dt, dD_dt


#====================================================================================================
# SIRDV Model differential equations Helper Function
#====================================================================================================
def sirdv_model(S_t, I_t, R_t, V_t, beta, gamma, mu, vac_rate):
    """'
    This function takes in a specified infection rate (beta), recovery rate (gamma), death rate (mu), vaccination rate (vac_rate)
    and the current number of susceptible (S), infected (I), recovered individuals (R),and vaccinated individuals (V) for a given day.
    It returns the rate of change in the number of susceptible individuals (dS/dt), change in the number
    of infected individuals (dI/dt),  change in the number of recovered individuals (dR/dt),
    change in the number of deaths (dD/dt), and change in the number of vaccinated individuals (dV/dt)  for that day.
    """
    dS_dt = -beta * ((S_t*I_t)/(S_t+I_t+R_t+ V_t)) - vac_rate*S_t#Change in number of susceptible individuals at day t
    dI_dt  = beta * ((S_t*I_t)/(S_t+I_t+R_t+V_t)) - gamma*I_t - mu*I_t #Change in number of infected people at day t
    dR_dt = gamma*I_t #Change in number of recovered individuals at day t
    dD_dt = mu*I_t
    dV_dt = vac_rate*S_t
    return dS_dt, dI_dt, dR_dt, dD_dt, dV_dt


#====================================================================================================
# Function to Run the Whole Simulation based on the model ("SIR", "SIRD" or "SIRDV") the user specifies
#====================================================================================================
# WRITE YOUR PSEUDOCODE HERE (I've started some of it for you within the function)

#I've written the parameters for the run_sim function. Tweak the function definition so the function
# simulates a "SIR" model for 100 days where S_0 = 997, I_0 = 3, R_0 = 0, V_0, beta=.4, gamma = .04,
# mu = 0, vac_rate = 0 BY DEFAULT.
# Hint: see Functions cheat sheet

def run_sim(S_0=997, I_0=3, R_0=0, beta=.4, gamma=.035, mu=0, vac_rate=0,days=100, model_choice="SIR"):
    """'
   This function runs the simulation using the specified parameters for the specified number of days.
   Returns the S,I,R,D,V arrays representing the number of individuals in each bucket per day.
    """

    #-----------------------------------------------------------------------
    #Initialize arrays of zeroes with length "days" to keep track of the S, I, R, D and V individuals for each day

    # Set initial number of susceptible (S), infected (I), recovered (R) & vaccinated (V) people on day 0
    # based on the parameters S_0, I_0, R_0
    #-----------------------------------------------------------------------
    # Note: The D and V arrays may not be used later in the function unless you specify the model_choice
    # parameter to be "SIRD" or "SIRDV"
    S = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    V = np.zeros(days)
    D = np.zeros(days)

    # Set initial values
    S[0], I[0], R[0] = S_0, I_0, R_0
    # Note: Deceased individuals who died from the disease at day 0 will be 0, so you don't need to assign D[0] to anything,
    # as the first element of the array is automatically 0


    # -----------------------------------------------------------------------
    # Based on the model choice ("SIR", "SIRD" or "SIRDV"), simulate the number of
    # susceptible (S), infected (I), recovered (R) and if relevant, the deceased (D) & vaccinated (V) people for all days
    # -----------------------------------------------------------------------
    for day in range(1, days): #for each day...
    #Get dS, dI, dR and if relevant, dD and dV to count up the number of S,I,R,D and V people for that day
    #based on the specified infection, recovery, death and vaccination rates and the # of S,I,R,D and V people the day before
    #Hint: use if, elif, else loop based on "model_choice" and call the helper functions you defined above

        #Get rates of change (dS, dI, dR, & if needed, dD, dV)
        if model_choice == "SIR":
            dS, dI, dR = sir_model(S_t=S[day - 1], I_t=I[day - 1], R_t=R[day - 1], beta=beta, gamma=gamma)
            dD = dV = 0
        elif model_choice == "SIRD":
            dS,dI,dR,dD = sird_model(S_t=S[day - 1], I_t=I[day - 1], R_t=R[day - 1], beta=beta, gamma=gamma,mu=mu)
            dV = 0
        else:
            dS,dI,dR,dD,dV = sirdv_model(S_t=S[day - 1], I_t=I[day - 1], R_t=R[day - 1], V_t = V[day-1],beta=beta, gamma=gamma,mu=mu,
                                         vac_rate=vac_rate)

        #Fill in relevant arrays w/ # of indv for that day based on the calculated dS, dI, dR, dD, dV
       #and the # of indv from the day before (day -1).

        S[day] = S[day-1] + dS
        I[day] = I[day-1] + dI
        R[day] = R[day-1] + dR
        D[day] = D[day-1] + dD
        V[day] = V[day-1] + dV

    # -----------------------------------------------------------------------
    # Return the final arrays
    # -----------------------------------------------------------------------
    return S, I, R, D, V