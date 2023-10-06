# 1. Theory

## 1.1 DS

### 1.1.1 Synchrone Schaltung

#### 1.1.1.1 clock skew/jitter
```






K1.
A list of t_logic,max has to be presented when we have multiple paths including those with feedbacks
Ob1-1. t_logic,max/t_logic,min

K2.
When determining value for clock skew, those t_logic,min with feedbacks should not be considered as an input, as feedbacks have no skew (FF connected to itself)

K3.
when listing the formel for timing condition with clock skew & jitter, first find the t_setup or t_hold and then do minus t_skew and plus 2*t_jitter for setup formula and do plus t_skew and plus 2*t_jitter for hold formula.
Ob3-1. setup_formula/hold_formula

K4.
When we reduce the clock frequency, we can correct the setup error but not the hold error, so the negative clock skew instead of positive ones can be therefore corrected, as positiv clock skew will relax the setup condition and damage the hold condition. 
Ob4-1. clock frequency

K5.
Clock skew can be caused by such as 7 factors shown in a figure.
Ob5-1. Clock Skew causes figure

K6.
Taktbaum can be used to reduce or make clock skew as small as possible.
--Ob1. taktbaum(clock tree)
K6-1. 
Ziel von Taktbaum ist, dass die Taktflanken an allen Registern gleichzeitig auftreten
K6-2.
the figure contains Registertakteingänge and Taktgenerator
--Ob1 Taktgenerator
K6-3.


K7.



```

### 1.2 





# 2. Projects


