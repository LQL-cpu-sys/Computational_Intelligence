import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

# 导入配置和模糊逻辑
from data import (
    MEAL_BUFFER_TIME, BATH_BUFFER_TIME, MAX_POWER, PENALTY_COEF,
    COMFORT_TEMP_HIGH, COMFORT_TEMP_LOW, DEVICE_DATA, ELECTRICITY_PRICE,
    TEMPERATURE_DATA_SUMMER as temperature_data 
)
from fuzzy_logic import create_fuzzy_system, get_power_level, get_price

def get_user_input():
    """获取用户输入（支持小数时间）"""
    while True:
        at_home = input("请问家里今天会有人吗？(YES/NO): ").strip().upper()
        if at_home in ("YES", "NO"):
            break
        print("请输入YES或NO")

    home_periods = []
    breakfast_time = lunch_time = dinner_time = bath_start_time = wake_up_time = None
    bath_duration = user_priority = None

    if at_home == "YES":
        print("请输入您在家的时间段（可以输入1-2个时间段，格式如8.5-12.25）：")
        for i in range(2):
            while True:
                try:
                    period = input(f"时间段{i+1}（直接回车结束输入）：").strip()
                    if not period:
                        if i == 0 and not home_periods:
                            print("至少需要输入一个时间段")
                            continue
                        break
                        
                    start, end = map(float, period.split("-"))
                    if not (0 <= start < 24 and 0 <= end < 24):
                        raise ValueError("时间必须在0-24之间")
                        
                    if start <= end:
                        home_periods.append((start, end))
                    else:
                        home_periods.append((start, 24))
                        home_periods.append((0, end))
                    break
                except ValueError as e:
                    print(f"输入错误：{e}")

        def get_float_time(prompt):
            """获取浮点数时间输入"""
            while True:
                try:
                    time = float(input(prompt))
                    if not 0 <= time < 24:
                        raise ValueError("时间必须在0-24之间")
                    return time
                except ValueError as e:
                    print(f"输入错误：{e}")

        # 获取用餐时间（支持小数）
        if home_periods:
            print("请输入用餐时间（支持小数如7.5表示7:30），不在家吃饭请直接回车跳过：")
            
            # 早餐时间
            breakfast_time = get_float_time("早餐时间（直接回车跳过）：") if input("需要输入早餐时间吗？(Y/N): ").strip().upper() == "Y" else None
            
            # 午餐时间 - 只有在家时间段包含午餐时间才询问
            lunch_in_home = any(start <= 12 <= end for start, end in home_periods)
            if lunch_in_home:
                lunch_time = get_float_time("午餐时间（直接回车跳过）：") if input("需要输入午餐时间吗？(Y/N): ").strip().upper() == "Y" else None
            else:
                print("您输入的时间段不包含午餐时间（12点左右），将不考虑午餐")
                lunch_time = None
                
            # 晚餐时间
            dinner_in_home = any(start <= 18 <= end for start, end in home_periods)
            if dinner_in_home:
                dinner_time = get_float_time("晚餐时间（直接回车跳过）：") if input("需要输入晚餐时间吗？(Y/N): ").strip().upper() == "Y" else None
            else:
                print("您输入的时间段不包含晚餐时间（18点左右），将不考虑晚餐")
                dinner_time = None
            
            wake_up_time = get_float_time("起床时间：")
            
            # 洗澡时间
            bath_start_time = get_float_time("洗澡开始时间（直接回车跳过）：") if input("需要输入洗澡时间吗？(Y/N): ").strip().upper() == "Y" else None
            if bath_start_time is not None:
                while True:
                    try:
                        bath_duration = float(input("洗澡时长（小时）："))
                        if bath_duration <= 0:
                            raise ValueError("必须大于0")
                        break
                    except ValueError as e:
                        print(f"输入错误：{e}")

        # 用户优先级
        while True:
            user_priority = input("优先级（0节省/1平衡/2舒适）：").strip()
            if user_priority in ("0", "1", "2"):
                user_priority = int(user_priority)
                break
            print("请输入0-2")

    return {
        "at_home": at_home == "YES",
        "home_periods": home_periods,
        "breakfast_time": breakfast_time,
        "lunch_time": lunch_time,
        "dinner_time": dinner_time,
        "bath_start_time": bath_start_time,
        "bath_duration": bath_duration,
        "user_priority": user_priority,
        "wake_up_time": wake_up_time
    }


def format_time(hours: float) -> str:
    """将小数小时转为HH:MM格式"""
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h:02d}:{m:02d}"

def generate_device_profiles():
    """生成设备配置"""
    devices = {}
    for device_name, data in DEVICE_DATA.items():
        devices[device_name] = {
            "power_options": data["功率等级"],
            "work_time_options": data["所需时间"]
        }
    return devices

def create_individual(devices, user_constraints, fuzzy_system):
    """创建个体调度方案"""
    individual = {}
    hourly_temps = [temp for _, temp in sorted(temperature_data, key=lambda x: x[0])]
    
    def get_price(hour: float) -> float:
        """获取指定小时的电价"""
        for start, end, price in ELECTRICITY_PRICE:
            if start <= hour < end:
                return price
        return ELECTRICITY_PRICE[-1][2]
    
    def get_power_level(device: str, current_time: float) -> str:
        """通过模糊系统获取功率等级"""
        hour_idx = int(current_time) % 24
        fuzzy_system.input['temperature'] = hourly_temps[hour_idx]
        fuzzy_system.input['price_level'] = get_price(current_time)
        fuzzy_system.input['user_preference'] = user_constraints["user_priority"]
        try:
            fuzzy_system.compute()
            pl_index = round(fuzzy_system.output['power_level'])
            return list(devices[device]["power_options"].keys())[min(pl_index, 2)]
        except:
            return "低"  # 默认低功率
    
    # 电饭煲处理
    if "电饭煲" in devices and user_constraints["at_home"]:
        meals = []
        if user_constraints["breakfast_time"] is not None:
            meals.append((user_constraints["breakfast_time"], "早餐"))
        if user_constraints["lunch_time"] is not None:
            meals.append((user_constraints["lunch_time"], "午餐"))
        if user_constraints["dinner_time"] is not None:
            meals.append((user_constraints["dinner_time"], "晚餐"))
        
        for meal_time, meal_name in meals:
            # 检查用餐时间是否在家的时间段内
            in_home_period = any(start <= meal_time <= end for start, end in user_constraints["home_periods"])
            if in_home_period:
                start = max(0, meal_time - random.uniform(*MEAL_BUFFER_TIME))
                power_level = get_power_level("电饭煲", start)
                duration = devices["电饭煲"]["work_time_options"][power_level]
                
                individual[f"电饭煲_{meal_name}"] = {
                    "start": start,
                    "end": start + duration,
                    "power_level": power_level,
                    "power": devices["电饭煲"]["power_options"][power_level],
                    "duration": int(duration * 60)
                }

    if "空调" in devices and user_constraints["at_home"]:
        # 找出所有需要空调的时段
        need_ac_periods = []
        
        for home_start, home_end in user_constraints["home_periods"]:
            if home_start >= home_end:
                periods = [(home_start, 24), (0, home_end)]
            else:
                periods = [(home_start, home_end)]
            
            for start, end in periods:
                current = start
                while current < end:
                    hour = int(current)
                    temp = hourly_temps[hour]
                    
                    # 判断是否需要空调（制冷或制热）
                    if temp > COMFORT_TEMP_HIGH or temp < COMFORT_TEMP_LOW:
                        ac_start = current
                        ac_end = min(current + 1, end)
                        
                        # 延长时段直到温度回到舒适区间或到达居家结束时间
                        while ac_end < end:
                            next_hour = int(ac_end)
                            next_temp = hourly_temps[next_hour]
                            if (next_temp > COMFORT_TEMP_HIGH) or (next_temp < COMFORT_TEMP_LOW):
                                ac_end += 1
                            else:
                                break
                        
                        # 确保不超过当前居家时段结束时间
                        ac_end = min(ac_end, end)
                        need_ac_periods.append((ac_start, ac_end))
                        current = ac_end
                    else:
                        current += 1
        
        # 合并重叠或相邻的时段
        if need_ac_periods:
            need_ac_periods.sort()
            merged_periods = [need_ac_periods[0]]
            for current_start, current_end in need_ac_periods[1:]:
                last_start, last_end = merged_periods[-1]
                if current_start <= last_end:
                    merged_periods[-1] = (last_start, max(last_end, current_end))
                else:
                    merged_periods.append((current_start, current_end))
        else:
            merged_periods = []
        
        # 为每个时段设置空调，确保不超过任何居家时段
        for start, end in merged_periods:
            # 找到所属的居家时段
            home_period = None
            for hp_start, hp_end in user_constraints["home_periods"]:
                if hp_start >= hp_end:  # 跨天情况
                    if (start >= hp_start or start < hp_end) or (end > hp_start or end <= hp_end):
                        home_period = (hp_start, hp_end)
                        break
                else:
                    if start >= hp_start and end <= hp_end:
                        home_period = (hp_start, hp_end)
                        break
            
            if home_period:
                # 调整结束时间不超过居家时段
                hp_start, hp_end = home_period
                if hp_start >= hp_end:  # 跨天情况
                    if end > hp_end and end <= hp_start:
                        end = hp_end
                else:
                    end = min(end, hp_end)
            
            mid_hour = (start + end) / 2
            temp = hourly_temps[int(mid_hour) % 24]
            price = get_price(mid_hour)
            
            fuzzy_system.input['temperature'] = temp
            fuzzy_system.input['price_level'] = price
            fuzzy_system.input['user_preference'] = user_constraints["user_priority"]
            
            try:
                fuzzy_system.compute()
                power_level_index = round(fuzzy_system.output['power_level'])
                power_levels = list(devices["空调"]["power_options"].keys())
                power_level = power_levels[min(power_level_index, len(power_levels)-1)]
            except:
                power_level = "低"
            
            duration = max(min(end - start, 24), 1)  # 确保不超过24小时
            power = devices["空调"]["power_options"][power_level]
            
            mode = "制热" if temp < COMFORT_TEMP_LOW else "制冷"
            
            individual[f"空调_{start}-{end}"] = {
                "start": start,
                "end": end,
                "power_level": power_level,
                "power": power,
                "duration": int(duration * 60),
                "mode": mode,
                "limited_by_home": end != merged_periods[merged_periods.index((start, end))][1]  # 标记是否被居家时间限制
            }

    # 洗衣机处理
    if "洗衣机" in devices and user_constraints["at_home"]:
        bath_time = user_constraints["bath_start_time"]
        wake_up = user_constraints["wake_up_time"]
        
        if bath_time is not None and wake_up is not None:
            min_start = (bath_time + 0.5) % 24
            max_end = (wake_up - 1) % 24
            
            if min_start < max_end:
                current_time = random.uniform(min_start, max_end - 0.5)
            else:
                if random.random() < 0.5:
                    current_time = random.uniform(min_start, 23.5)
                else:
                    current_time = random.uniform(0, max_end - 0.5)

            power_level = get_power_level("洗衣机", current_time)
            work_hours = devices["洗衣机"]["work_time_options"][power_level]
            
            individual["洗衣机"] = {
                "start": current_time,
                "end": (current_time + work_hours) % 24,
                "power_level": power_level,
                "power": devices["洗衣机"]["power_options"][power_level],
                "duration": int(work_hours * 60)
            }

    # 热水器处理
    if "热水器" in devices and user_constraints["at_home"]:
        bath_time = user_constraints["bath_start_time"]
        if bath_time is not None:
            start_time = max(0, bath_time - random.uniform(*BATH_BUFFER_TIME))
            power_level = get_power_level("热水器", start_time)
            work_hours = devices["热水器"]["work_time_options"][power_level]
            
            individual["热水器"] = {
                "start": start_time,
                "end": (start_time + work_hours) % 24,
                "power_level": power_level,
                "power": devices["热水器"]["power_options"][power_level],
                "duration": int(work_hours * 60)
            }

    # 冰箱处理
    if "冰箱" in devices:
        # 将温度数据转换为按小时排序的列表
        hourly_temps = [temp for _, temp in sorted(temperature_data, key=lambda x: x[0])]
        
        # 默认设置为低功率
        power_level = "低"
        power = devices["冰箱"]["power_options"]["低"]
        
        # 计算24小时平均功率水平
        power_levels = list(devices["冰箱"]["power_options"].keys())
        total_power = 0
        
        for hour in range(24):
            price = next(p for s, e, p in ELECTRICITY_PRICE if s <= hour < e)
            
            fuzzy_system.input['temperature'] = hourly_temps[hour]
            fuzzy_system.input['price_level'] = price
            fuzzy_system.input['user_preference'] = user_constraints["user_priority"]
            fuzzy_system.compute()
            
            # 获取功率水平
            level_index = min(max(round(fuzzy_system.output['power_level']), 0), len(power_levels)-1)
            total_power += devices["冰箱"]["power_options"][power_levels[level_index]]
        
        # 计算平均功率并选择最接近的功率水平
        avg_power = total_power / 24
        power_level = min(devices["冰箱"]["power_options"].keys(),
                        key=lambda x: abs(devices["冰箱"]["power_options"][x] - avg_power))
        power = devices["冰箱"]["power_options"][power_level]

        individual["冰箱"] = {
            "start": 0,
            "end": 24,
            "power_level": power_level,
            "power": power,
            "duration": 1440
        }
    return individual

#适应度计算
def calculate_fitness(individual, devices, user_constraints):
    electricity_cost = 0.0
    hourly_power = defaultdict(float)
    comfort_penalty = 0.0
    
    # 创建模糊系统
    fuzzy_system = create_fuzzy_system(
        [temp for _, temp in sorted(temperature_data, key=lambda x: x[0])],
        user_constraints["user_priority"]
    )
    
    # 预生成查询表
    price_lookup = {hour: next((p for s,e,p in ELECTRICITY_PRICE if s<=hour<e), 0) 
                   for hour in range(24)}
    temp_lookup = {hour: temp for hour, temp in sorted(temperature_data, key=lambda x: x[0])}
    
    # 1. 计算电费和各时段功率
    for device, params in individual.items():
        start = params["start"] % 24  # 规范化时间
        end = params["end"] % 24
        if start >= end:  # 处理跨天设备
            end += 24
        
        current = start
        while current < end:
            hour = int(current) % 24
            runtime = min(1 - (current % 1), end - current)
            electricity_cost += (params["power"] / 1000) * runtime * price_lookup[hour]
            hourly_power[hour] += params["power"] * runtime
            current += runtime
    
    # 2. 计算舒适度惩罚（基于模糊逻辑）
    if user_constraints["at_home"]:
        # 生成在家时段掩码
        home_hours = set()
        for start, end in user_constraints["home_periods"]:
            if start < end:
                home_hours.update(range(int(start), int(end)+1))
            else:
                home_hours.update(range(int(start), 24))
                home_hours.update(range(0, int(end)+1))
        
        for hour in range(24):
            if hour not in home_hours:
                continue
                
            temp = temp_lookup[hour]
            price = price_lookup[hour]
            
            # 2.1 计算理想功率
            fuzzy_system.input['temperature'] = temp
            fuzzy_system.input['price_level'] = price
            fuzzy_system.input['user_preference'] = user_constraints["user_priority"]
            fuzzy_system.compute()
            ideal_power_level = round(fuzzy_system.output['power_level'])  # 0-2
            
            # 2.2 获取实际功率
            actual_power_level = 0 
            for dev_name, params in individual.items():
                if dev_name.startswith("空调_"):
                    dev_start = params["start"] % 24
                    dev_end = params["end"] % 24
                    if dev_start <= hour < dev_end or (dev_start > dev_end and (hour >= dev_start or hour < dev_end)):
                        # 功率等级映射：低->0, 中->1, 高->2
                        actual_power_level = {"低":0, "中":1, "高":2}[params["power_level"]]
                        break
            
            # 2.3 计算惩罚
            temp_diff = max(COMFORT_TEMP_LOW - temp, temp - COMFORT_TEMP_HIGH, 0)
            level_diff = abs(ideal_power_level - actual_power_level)
            
            # 惩罚公式：温度差×等级差×优先级权重
            comfort_penalty += temp_diff * level_diff * (user_constraints["user_priority"] + 1) * 0.1
    
    # 3. 功率超限惩罚
    peak_power = max(hourly_power.values(), default=0)
    power_penalty = max(0, peak_power - MAX_POWER) * PENALTY_COEF
    
    # 4. 综合适应度计算（越小越好）
    total_cost = electricity_cost + power_penalty + comfort_penalty
    fitness = 1.0 / (total_cost + 0.1)  # 避免除零
    
    return {
        "fitness": fitness,
        "electricity_cost": round(electricity_cost, 2),
        "power_penalty": round(power_penalty, 2),
        "comfort_penalty": round(comfort_penalty, 2),
        "hourly_power": dict(hourly_power),
        "peak_power": round(peak_power, 2)
    }

#粒子群算法
class Particle:
    def __init__(self, devices, user_constraints, fuzzy_system):
        """粒子初始化"""
        self.position = create_individual(devices, user_constraints, fuzzy_system)
        self.velocity = self._initialize_velocity(devices, user_constraints)
        self.best_position = self.position.copy()
        self.best_fitness = -float('inf')
        self.fitness = -float('inf')
        self.w_max = 1.2  # 增加最大惯性权重
        self.w_min = 0.2  # 减少最小惯性权重
        self.c1_init = 2.0  # 个体学习因子
        self.c2_init = 2.0  # 社会学习因子
    
    def _initialize_velocity(self, devices, user_constraints):
        """初始化粒子速度"""
        velocity = {}
        
        # 电饭煲速度
        if "电饭煲" in devices and user_constraints["at_home"]:
            meals = ["早餐", "午餐", "晚餐"]
            for meal in meals:
                key = f"电饭煲_{meal}"
                velocity[key] = {
                    "start": random.uniform(-1, 1),
                    "power_level": random.uniform(-0.5, 0.5)
                }
        
        # 洗衣机速度
        if "洗衣机" in devices and user_constraints["at_home"]:
            velocity["洗衣机"] = {
                "start": random.uniform(-1, 1),
                "power_level": random.uniform(-0.5, 0.5)
            }
        
        # 热水器速度
        if "热水器" in devices and user_constraints["at_home"]:
            velocity["热水器"] = {
                "start": random.uniform(-1, 1),
                "power_level": random.uniform(-0.5, 0.5)
            }
        
        return velocity
    
    def update_position(self, devices, user_constraints):
        """更新粒子位置"""
        new_position = {}
    
        # 1. 更新电饭煲位置
        if "电饭煲" in devices and user_constraints["at_home"]:
            meals = []
            if user_constraints["breakfast_time"] is not None:
                meals.append(("早餐", user_constraints["breakfast_time"]))
            if user_constraints["lunch_time"] is not None:
                meals.append(("午餐", user_constraints["lunch_time"]))
            if user_constraints["dinner_time"] is not None:
                meals.append(("晚餐", user_constraints["dinner_time"]))
            
            for meal_name, meal_time in meals:
                key = f"电饭煲_{meal_name}"
                if key in self.position:
                    original = self.position[key]
                    vel = self.velocity.get(key, {"start": 0, "power_level": 0})
                    
                    # 更灵活的时间更新
                    new_start = original["start"] + vel["start"]
                    max_start = meal_time - 0.1  # 最少提前6分钟
                    min_start = meal_time - 2.0  # 最多提前2小时
                    new_start = max(min_start, min(new_start, max_start))
                    
                    # 功率等级更新
                    power_levels = list(devices["电饭煲"]["power_options"].keys())
                    current_idx = power_levels.index(original["power_level"])
                    new_idx = current_idx + round(vel["power_level"])
                    new_idx = max(0, min(new_idx, len(power_levels)-1))
                    new_power_level = power_levels[new_idx]
                    
                    # 更新位置
                    work_time = devices["电饭煲"]["work_time_options"][new_power_level]
                    new_position[key] = {
                        "start": new_start,
                        "end": new_start + work_time,
                        "power_level": new_power_level,
                        "power": devices["电饭煲"]["power_options"][new_power_level],
                        "duration": int(work_time * 60)
                    }
        
        # 2. 更新洗衣机位置（更灵活的时间范围）
        if "洗衣机" in devices and user_constraints["at_home"]:
            if "洗衣机" in self.position and user_constraints["bath_start_time"] is not None and user_constraints["wake_up_time"] is not None:
                original = self.position["洗衣机"]
                vel = self.velocity.get("洗衣机", {"start": 0, "power_level": 0})
                
                # 更灵活的时间范围
                new_start = original["start"] + vel["start"]
                min_start = (user_constraints["bath_start_time"] + 0.1) % 24  # 洗澡后6分钟
                max_end = (user_constraints["wake_up_time"] - 0.1) % 24  # 起床前6分钟
                
                if min_start < max_end:
                    new_start = max(min_start, min(new_start, max_end))
                else:
                    if new_start > min_start:
                        new_start = max(min_start, min(new_start, 23.9))
                    else:
                        new_start = max(0, min(new_start, max_end))
                
                # 功率等级更新
                power_levels = list(devices["洗衣机"]["power_options"].keys())
                current_idx = power_levels.index(original["power_level"])
                new_idx = current_idx + round(vel["power_level"])
                new_idx = max(0, min(new_idx, len(power_levels)-1))
                new_power_level = power_levels[new_idx]
                
                # 更新位置
                work_time = devices["洗衣机"]["work_time_options"][new_power_level]
                new_position["洗衣机"] = {
                    "start": new_start,
                    "end": (new_start + work_time) % 24,
                    "power_level": new_power_level,
                    "power": devices["洗衣机"]["power_options"][new_power_level],
                    "duration": int(work_time * 60)
                }
        
        # 热水器位置更新
        if "热水器" in devices and user_constraints["at_home"]:
            if "热水器" in self.position and user_constraints["bath_start_time"] is not None:
                original = self.position["热水器"]
                vel = self.velocity.get("热水器", {"start": 0, "power_level": 0})
                
                # 更新开始时间
                new_start = original["start"] + vel["start"]
                end_time = user_constraints["bath_start_time"] - random.uniform(*BATH_BUFFER_TIME)
                new_start = max(0, min(new_start, end_time - 0.1))
                
                # 更新功率等级
                power_levels = list(devices["热水器"]["power_options"].keys())
                current_idx = power_levels.index(original["power_level"])
                new_idx = round(current_idx + vel["power_level"])
                new_idx = max(0, min(new_idx, len(power_levels)-1))
                new_power_level = power_levels[new_idx]
                
                # 创建新位置
                work_time = devices["热水器"]["work_time_options"][new_power_level]
                new_position["热水器"] = {
                    "start": new_start,
                    "end": (new_start + work_time) % 24,
                    "power_level": new_power_level,
                    "power": devices["热水器"]["power_options"][new_power_level],
                    "duration": int(work_time * 60)
                }
        
        # 空调和冰箱保持不变
        for key in self.position:
            if key.startswith("空调_") or key == "冰箱":
                new_position[key] = self.position[key].copy()
        
        self.position = new_position
    
    def update_velocity(self, global_best_position, gen, max_gen):
        # 动态调整惯性权重和学习因子
        w = self.w_max - (self.w_max - self.w_min) * (gen / max_gen)
        c1 = self.c1_init * (1 - 0.5 * gen/max_gen)  # 逐渐减少个体认知
        c2 = self.c2_init * (1 + 0.5 * gen/max_gen)  # 逐渐增加社会认知
        
        for key in self.position:
            if key not in self.velocity:
                # 为新设备初始化速度
                self.velocity[key] = {
                    "start": random.uniform(-1, 1),
                    "power_level": random.uniform(-0.5, 0.5)
                }
            
            # 确保全局最优位置中有当前设备
            if key not in global_best_position:
                continue
                
            r1, r2 = random.random(), random.random()
            
            # 更新时间速度
            vel_start = (w * self.velocity.get(key, {}).get("start", 0) + 
                        c1 * r1 * (self.best_position[key]["start"] - self.position[key]["start"]) + 
                        c2 * r2 * (global_best_position[key]["start"] - self.position[key]["start"]))
            
            # 更新功率等级速度
            device_name = key.split("_")[0]
            power_levels = list(DEVICE_DATA[device_name]["功率等级"].keys())
            current_idx = power_levels.index(self.position[key]["power_level"])
            best_idx = power_levels.index(self.best_position[key]["power_level"])
            global_idx = power_levels.index(global_best_position[key]["power_level"])
            
            vel_power = (w * self.velocity.get(key, {}).get("power_level", 0) + 
                        c1 * r1 * (best_idx - current_idx) + 
                        c2 * r2 * (global_idx - current_idx))
            
            # 限制速度范围
            self.velocity[key] = {
                "start": max(-2, min(2, vel_start)),
                "power_level": max(-1.5, min(1.5, vel_power))
            }

def particle_swarm_optimization(devices, user_constraints, population_size=30, generations=50):
    """粒子群优化算法主函数"""
    hourly_temps = [temp for _, temp in sorted(temperature_data, key=lambda x: x[0])]
    fuzzy_system = create_fuzzy_system(hourly_temps, user_constraints["user_priority"])
    
    # 初始化粒子群
    particles = [Particle(devices, user_constraints, fuzzy_system) for _ in range(population_size)]
    global_best_position = None
    global_best_fitness = -float('inf')
    best_history = []
    
    for gen in range(generations):
        # 评估粒子
        for particle in particles:
            fitness_result = calculate_fitness(particle.position, devices, user_constraints)
            particle.fitness = fitness_result["fitness"]
            
            # 更新个体最佳
            if particle.fitness > particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position.copy()
            
            # 更新全局最佳
            if particle.fitness > global_best_fitness:
                global_best_fitness = particle.fitness
                global_best_position = particle.position.copy()
        
        best_history.append(global_best_fitness)
        
        # 输出当前最优结果
        best_result = calculate_fitness(global_best_position, devices, user_constraints)
        print(f"Generation {gen}: Fitness={best_result['fitness']:.4f} "
              f"Cost={best_result['electricity_cost']:.2f}+{best_result['power_penalty']:.2f} "
              f"Comfort={best_result['comfort_penalty']:.2f}")
        
        # 更新粒子速度和位置
        for particle in particles:
            particle.update_velocity(global_best_position, gen, generations)
            particle.update_position(devices, user_constraints)
    
    return {
        "schedule": global_best_position,
        "fitness_history": best_history,
        "statistics": {
            "electricity_cost": best_result["electricity_cost"],
            "power_penalty": best_result["power_penalty"],
            "comfort_penalty": best_result["comfort_penalty"],
            "peak_power": max(best_result["hourly_power"].values())
        }
    }

def print_schedule(schedule):
    """打印调度方案"""
    print("\n=== 最优调度方案 ===")
    print(f"{'设备':<10}{'时段':<15}{'功率':<8}{'模式':<10}{'时长':<10}")
    print("-"*50)
    
    for name, params in sorted(schedule.items(), key=lambda x: x[1]["start"]):
        dev_name = name.split("_")[0]
        if dev_name == "电饭煲" and "_" in name:
            meal_name = name.split("_")[1]
            dev_name = f"电饭煲({meal_name})"
        
        mode = params.get("mode", "")
        print(
            f"{dev_name:<10} "
            f"{format_time(params['start'])}-{format_time(params['end']):<15} "
            f"{params['power']:<8} "
            f"{mode if mode else params['power_level']:<8} "
            f"{params['duration']}分钟"
        )


if __name__ == "__main__":
    # 1. 获取用户输入
    user_input = get_user_input()
    # 2. 生成设备配置
    devices = generate_device_profiles()
    # 3. 运行遗传算法
    print("\n正在优化调度方案...")
    result = particle_swarm_optimization(devices, user_input)
    # 4. 输出结果
    print_schedule(result["schedule"])
    print("\n统计信息:")
    print(f"总电费: {result['statistics']['electricity_cost']:.2f}元")
    print(f"功率惩罚: {result['statistics']['power_penalty']:.2f}元")
    print(f"峰值功率: {result['statistics']['peak_power']:.0f}W")