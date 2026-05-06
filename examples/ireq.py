from pythermalcomfort.models import ireq

result = ireq(
    tdb=-15.0,
    tr=-15.0,
    v=2.0,
    rh=55.0,
    met=175.0 / 58.15,
    clo=2.8,
    p=50.0,
    walk_sp=1.1,
)

print(result)
