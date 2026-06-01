import time
from multiprocessing import Process, Value

import axis_satuei_rgb_only_concat as axis_mod

# -------- classes --------
gf2000 = axis_mod.gf2000
SC800IM700_1 = axis_mod.SC800IM700_1
SC800IM700_2 = axis_mod.SC800IM700_2
SC800IM700_3 = axis_mod.SC800IM700_3
SC800IM700_4 = axis_mod.SC800IM700_4

# -------- ports --------
XY_PORT_1 = "COM8"
XY_PORT_2 = "COM9"
XY_PORT_3 = "COM10"
XY_PORT_4 = "COM12"
XY_ADDRESS = 0x2A

Z_PORT = "COM15"

# -------- constants --------
N2GF = 101.972  # gf -> N


def main():
    # shared memory
    normal_force = Value("f", 0.0)
    shear_force1 = Value("f", 0.0)
    shear_force2 = Value("f", 0.0)
    shear_force3 = Value("f", 0.0)
    shear_force4 = Value("f", 0.0)
    ser_flag = Value("b", True)

    count1 = Value("i", 0)
    count2 = Value("i", 0)
    count3 = Value("i", 0)
    count4 = Value("i", 0)

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

    # subprocesses
    sub_z = Process(target=gf2000.sub_loop, args=[Z_PORT, ser_flag, normal_force])

    sub_xy1 = Process(target=SC800IM700_1.sub_loop,
                      args=[XY_PORT_1, XY_ADDRESS, ser_flag, shear_force1, count1])
    sub_xy2 = Process(target=SC800IM700_2.sub_loop,
                      args=[XY_PORT_2, XY_ADDRESS, ser_flag, shear_force2, count2])
    sub_xy3 = Process(target=SC800IM700_3.sub_loop,
                      args=[XY_PORT_3, XY_ADDRESS, ser_flag, shear_force3, count3])
    sub_xy4 = Process(target=SC800IM700_4.sub_loop,
                      args=[XY_PORT_4, XY_ADDRESS, ser_flag, shear_force4, count4])

    sub_z.start()
    sub_xy1.start()
    sub_xy2.start()
    sub_xy3.start()
    sub_xy4.start()

    # zero offsets
    fr0 = 0.0
    ff0 = 0.0
    z0 = 0.0

    print("\n=== センサ確認開始 ===")
    print("使い方:")
    print("  z + Enter : 現在値をゼロ点にする")
    print("  q + Enter : 終了")
    print("  何も入力せず Enter : そのまま継続\n")

    try:
        while True:
            time.sleep(0.2)

            # raw values
            s1 = float(shear_force1.value)
            s2 = float(shear_force2.value)
            s3 = float(shear_force3.value)
            s4 = float(shear_force4.value)
            z_raw = float(normal_force.value)

            # converted values
            fz_n = z_raw / N2GF - z0
            fx = (s1 - s3) - fr0
            fy = (s2 - s4) - ff0

            print("\033[2J\033[H", end="")  # 画面クリア
            print("=== 5センサ確認画面 ===")
            print(f"raw normal(gf): {z_raw:8.2f}")
            print(f"raw shear1(gf): {s1:8.2f}")
            print(f"raw shear2(gf): {s2:8.2f}")
            print(f"raw shear3(gf): {s3:8.2f}")
            print(f"raw shear4(gf): {s4:8.2f}")
            print("-" * 36)
            print(f"Fz(N): {fz_n:8.3f}")
            print(f"Fx(gf): {fx:8.3f}")
            print(f"Fy(gf): {fy:8.3f}")
            print("-" * 36)
            print("z=ゼロ点取得, q=終了, Enter=継続")

            # 非ブロッキングっぽく簡易入力
            start = time.time()
            cmd = ""
            while time.time() - start < 0.8:
                if _stdin_has_data():
                    cmd = input().strip().lower()
                    break
                time.sleep(0.05)

            if cmd == "z":
                fr0 = s1 - s3
                ff0 = s2 - s4
                z0 = z_raw / N2GF
                print("ゼロ点を更新しました")
                time.sleep(0.8)

            elif cmd == "q":
                break

    finally:
        print("\n終了処理中...")
        ser_flag.value = False

        sub_z.join(timeout=2.0)
        sub_xy1.join(timeout=2.0)
        sub_xy2.join(timeout=2.0)
        sub_xy3.join(timeout=2.0)
        sub_xy4.join(timeout=2.0)

        print("終了しました")


def _stdin_has_data():
    """
    Windowsコンソール向け簡易判定
    """
    try:
        import msvcrt
        return msvcrt.kbhit()
    except ImportError:
        return False


if __name__ == "__main__":
    main()