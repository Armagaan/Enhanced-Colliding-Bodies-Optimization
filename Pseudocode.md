# Pseucodee for ECOB

```
create cb_matrix and allot random positions in it.
(the first column of cb_matrix stores mass of each colliding body, other columns represent dimensions.)

iterate:
    get mass
    sort in descending order based on mass
    if iter == 0:
        setup memory vector
    else:
        replace bottom X entries with memory entries
    save top x entries in memory
    sort cb again
    plot (optional)
    set initial velocities
    calculate new velocities
    calculate new positions
    escape minima by incorporating Pro
    update epsilon
    
return cb_matrix
```
