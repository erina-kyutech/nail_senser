# -*- coding: utf-8 -*-
import time
import cv2
from multiprocessing import Process, Value

import axis_satuei_rgb_only_concat as axis_mod

# -------- classes --------
gf2000 = axis_mod.gf2000
SC800IM700_1 = axis_mod.SC800IM700_1
SC800IM700_2 = axis_mod.SC800IM700_2
SC800IM700_3 = axis_mod.SC800IM700_3
SC800IM700_4 = axis_mod.SC800IM700_4

# -------- ports --------
# 必要に応じてここは直してね
XY_PORT_1 = "COM8"
XY_PORT_2 = "COM9"
XY_PORT_3 = "COM10"
XY_PORT_4 = "COM12"
XY_ADDRESS = 0x2A
Z_PORT = "COM21"

# -------- constants --------
N2GF = 101.972  # gf -> N


class SinglePushTester:
    def __init__(self):
        self.normal_force = Value("f", 0.0)
        self.shear_force1 = Value("f", 0.0)
        self.shear_force2 = Value("f", 0.0)
        self.shear_force3 = Value("f", 0.0)
        self.shear_force4 = Value("f", 0.0)
        self.ser_flag = Value("b", True)

        self.count1 = Value("i", 0)
        self.count2 = Value("i", 0)
        self.count3 = Value("i", 0)
        self.count4 = Value("i", 0)

        # init sensors
        print("=== センサ初期化中 ===")
        shear_loadcell_1 = SC800IM700_1(XY_PORT_1, XY_ADDRESS)
        shear_loadcell_1.power_on()
        shear_loadcell_1.sub_ready()

        shear_loadcell_2 = SC800IM700_2(XY_PORT_2, XY_ADDRESS)
        shear_loadcell_2.power_on()
        shear_loadcell_2.sub_ready()

        shear_loadcell_3 = SC800IM700_3(XY_PORT_3, XY_ADDRESS)
        shear_loadcell_3.power_on()
        shear_loadcell_3.sub_ready()

        shear_loadcell_4 = SC800IM700_4(XY_PORT_4, XY_ADDRESS)
        shear_loadcell_4.power_on()
        shear_loadcell_4.sub_ready()

        normal_loadcell = gf2000(Z_PORT)
        normal_loadcell.sub_ready()

        self.sub_z = Process(target=gf2000.sub_loop, args=[Z_PORT, self.ser_flag, self.normal_force])

        self.sub_xy1 = Process(
            target=SC800IM700_1.sub_loop,
            args=[XY_PORT_1, XY_ADDRESS, self.ser_flag, self.shear_force1, self.count1]
        )
        self.sub_xy2 = Process(
            target=SC800IM700_2.sub_loop,
            args=[XY_PORT_2, XY_ADDRESS, self.ser_flag, self.shear_force2, self.count2]
        )
        self.sub_xy3 = Process(
            target=SC800IM700_3.sub_loop,
            args=[XY_PORT_3, XY_ADDRESS, self.ser_flag, self.shear_force3, self.count3]
        )
        self.sub_xy4 = Process(
            target=SC800IM700_4.sub_loop,
            args=[XY_PORT_4, XY_ADDRESS, self.ser_flag, self.shear_force4, self.count4]
        )

        self.sub_z.start()
        self.sub_xy1.start()
        self.sub_xy2.start()
        self.sub_xy3.start()
        self.sub_xy4.start()

        self.steps = [
            ("1", "センサ1だけを軽く押してください", "s1"),
            ("2", "センサ2だけを軽く押してください", "s2"),
            ("3", "センサ3だけを軽く押してください", "s3"),
            ("4", "センサ4だけを軽く押してください", "s4"),
            ("5", "垂直センサだけを軽く押してください", "z"),
        ]
        self.step_index = 0

        self.base = None
        self.measure_start = None
        self.measure_duration = 3.0  # 秒
        self.result_text = "spaceで計測開始 / nで次へ / ESCで終了"

        cv2.namedWindow("single_push_test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("single_push_test", 1000, 650)

    def get_raw(self):
        s1 = float(self.shear_force1.value)
        s2 = float(self.shear_force2.value)
        s3 = float(self.shear_force3.value)
        s4 = float(self.shear_force4.value)
        z = float(self.normal_force.value)
        return {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "z": z}

    def start_measure(self):
        self.base = self.get_raw()
        self.measure_start = time.time()
        self.max_abs_delta = {"s1": 0.0, "s2": 0.0, "s3": 0.0, "s4": 0.0, "z": 0.0}
        self.last_signed_delta = {"s1": 0.0, "s2": 0.0, "s3": 0.0, "s4": 0.0, "z": 0.0}
        self.result_text = "計測中... 指定されたセンサだけ押して"

    def update_measure(self):
        raw = self.get_raw()
        for k in self.max_abs_delta.keys():
            d = raw[k] - self.base[k]
            self.last_signed_delta[k] = d
            if abs(d) > self.max_abs_delta[k]:
                self.max_abs_delta[k] = abs(d)

        if time.time() - self.measure_start >= self.measure_duration:
            expected_key = self.steps[self.step_index][2]
            vals = self.max_delta

            sorted_vals = sorted(vals.items(), key=lambda x: x[1], reverse=True)
            top_key, top_val = sorted_vals[0]
            second_key, second_val = sorted_vals[1]

            ok = (top_key == expected_key)

            self.result_text = (
                f"終了: expected={expected_key}, top={top_key} ({top_val:.2f}), "
                f"2nd={second_key} ({second_val:.2f}) -> "
                + ("OK" if ok else "NG")
            )
            self.measure_start = None

    def draw(self):
        raw = self.get_raw()
        img = 255 * 0 * cv2.UMat(650, 1000, cv2.CV_8UC3).get()
        img[:] = (20, 20, 20)

        def put(text, x, y, color=(255, 255, 255), scale=0.8, thick=2):
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

        step_no, instruction, expected_key = self.steps[self.step_index]

        put("単独押し試験", 30, 40, (0, 255, 255), 1.0, 2)
        put(f"Step {step_no}/5", 30, 80, (255, 255, 255))
        put(instruction, 30, 120, (0, 255, 0))

        put("raw values", 30, 180, (255, 255, 0))
        put(f"s1(gf): {raw['s1']:8.2f}", 50, 220)
        put(f"s2(gf): {raw['s2']:8.2f}", 50, 260)
        put(f"s3(gf): {raw['s3']:8.2f}", 50, 300)
        put(f"s4(gf): {raw['s4']:8.2f}", 50, 340)
        put(f"z (gf): {raw['z']:8.2f}", 50, 380)
        put(f"z (N) : {raw['z']/N2GF:8.3f}", 50, 420, (255, 200, 0))

        put("操作: space=計測開始, n=次へ, ESC=終了", 30, 500, (180, 180, 180))
        put(self.result_text, 30, 550, (0, 200, 255))

        if hasattr(self, "max_delta") and self.measure_start is not None:
            put("max delta during measurement", 500, 180, (255, 255, 0))
            put(f"s1: {self.max_delta['s1']:8.2f}", 520, 220)
            put(f"s2: {self.max_delta['s2']:8.2f}", 520, 260)
            put(f"s3: {self.max_delta['s3']:8.2f}", 520, 300)
            put(f"s4: {self.max_delta['s4']:8.2f}", 520, 340)
            put(f"z : {self.max_delta['z']:8.2f}", 520, 380)

        cv2.imshow("single_push_test", img)

    def loop(self):
        print("=== 単独押し試験開始 ===")
        print("space: 計測開始, n: 次へ, ESC: 終了")

        try:
            while True:
                if self.measure_start is not None:
                    self.update_measure()

                self.draw()

                key = cv2.waitKey(50) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord(" "):
                    if self.measure_start is None:
                        self.start_measure()
                elif key == ord("n"):
                    if self.step_index < len(self.steps) - 1:
                        self.step_index += 1
                        self.measure_start = None
                        self.result_text = "spaceで計測開始 / nで次へ / ESCで終了"
                    else:
                        self.result_text = "最後のステップです"

        finally:
            print("終了処理中...")
            self.ser_flag.value = False

            self.sub_z.join(timeout=2.0)
            self.sub_xy1.join(timeout=2.0)
            self.sub_xy2.join(timeout=2.0)
            self.sub_xy3.join(timeout=2.0)
            self.sub_xy4.join(timeout=2.0)

            cv2.destroyAllWindows()
            print("終了しました")


if __name__ == "__main__":
    tester = SinglePushTester()
    tester.loop()