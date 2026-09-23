import mujoco

model = mujoco.MjModel.from_xml_path("leg.xml")
data = mujoco.MjData(model)
torso = model.body("torso").id
SETTLE, PUSH, WATCH = 1.0, 0.1, 3.0  # seconds


def survives(force):
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    while data.time < SETTLE + PUSH + WATCH:
        pushing = SETTLE <= data.time < SETTLE + PUSH
        data.xfrc_applied[torso, 0] = force if pushing else 0.0
        mujoco.mj_step(model, data)
    return data.xpos[torso, 2] > 0.7  # hip still high = still standing


for direction, name in [(+1, "forward"), (-1, "backward")]:
    last_ok = 0
    for f in range(10, 500, 10):
        if survives(direction * f):
            last_ok = f
        else:
            break
    print(f"Survives a {last_ok} N push {name}")
