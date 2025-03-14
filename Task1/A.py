"""
Forward (visualized)
Z=pre-activation value
W=weights
b=bias
A=activation function
X=input

 X  → [W1, b1] → Z1 → Activation → A1  
         ↓  
       [W2, b2] → Z2 → Activation → A2  
         ↓  
       [W3, b3] → Z3 → Activation → A3  
         ↓  
       [W4, b4] → Z4 → Activation → Ŷ (final output)

       
Backward (visualized)

Loss L ← [dW4, db4] ← dZ4 ← dA3  
         ↑  
        [dW3, db3] ← dZ3 ← dA2  
         ↑  
        [dW2, db2] ← dZ2 ← dA1  
         ↑  
        [dW1, db1] ← dZ1 ← dX
 
"""   
3