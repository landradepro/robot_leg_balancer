import time
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("leg.xml")
data = mujoco.MjData(model)  # ctrl = 0 -> all joints held straight

steps_per_frame = int(1 / 60 / model.opt.timestep)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t0 = time.time()
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(max(0.0, steps_per_frame * model.opt.timestep - (time.time() - t0)))
