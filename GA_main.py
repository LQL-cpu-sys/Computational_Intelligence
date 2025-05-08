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

        # 获取用餐时间
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
        
        # 为每个时段设置空调
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
                "limited_by_home": end != merged_periods[merged_periods.index((start, end))][1]  
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

def calculate_fitness(individual, devices, user_constraints):
    """改进的适应度计算，加入舒适度惩罚"""
    electricity_cost = 0.0
    hourly_power = defaultdict(float)
    comfort_penalty = 0.0
    
    # 预生成电价和温度查询表
    price_lookup = {}
    temp_lookup = {}
    for hour in range(24):
        for start, end, price in ELECTRICITY_PRICE:
            if start <= hour < end:
                price_lookup[hour] = price
                break
        temp_lookup[hour] = temperature_data[hour][1]
    
    for device, params in individual.items():
        start = params["start"]
        end = params["end"]
        power = params["power"]
        
        # 处理跨天情况
        if start >= end:
            continue
            
        # 计算每个小时的用电量
        current = start
        while current < end:
            hour = int(current) % 24
            runtime = min(1 - (current % 1), end - current)
            
            electricity_cost += (power / 1000) * runtime * price_lookup[hour]
            hourly_power[hour] += power * runtime
            current += runtime
    
    # 计算舒适度惩罚
    if user_constraints["at_home"]:
        for hour in range(24):
            temp = temp_lookup[hour]
            user_home = any(
                (start <= hour < end) or 
                (start > end and (hour >= start or hour < end))  # 处理跨天情况
                for start, end in user_constraints["home_periods"]
            )
            
            if user_home and (temp > COMFORT_TEMP_HIGH or temp < COMFORT_TEMP_LOW):
                ac_running = False
                for dev_name, params in individual.items():
                    if dev_name.startswith("空调_"):
                        dev_start = params["start"]
                        dev_end = params["end"]
                        
                        # 改进的跨天检查
                        if dev_start < dev_end:
                            if dev_start <= hour < dev_end:
                                ac_running = True
                                break
                        else:
                            if hour >= dev_start or hour < dev_end:
                                ac_running = True
                                break
                
                if not ac_running:
                    # 根据温度超出程度计算惩罚
                    if temp > COMFORT_TEMP_HIGH:
                        comfort_penalty += (temp - COMFORT_TEMP_HIGH) * 0.2 * (user_constraints["user_priority"] + 1)
                    else:
                        comfort_penalty += (COMFORT_TEMP_LOW - temp) * 0.2 * (user_constraints["user_priority"] + 1)
    
    # 功率惩罚
    peak_power = max(hourly_power.values(), default=0)
    if peak_power > MAX_POWER:
        power_penalty = (peak_power - MAX_POWER) * PENALTY_COEF
    else:
        power_penalty = 0
    
    # 平衡电费和舒适度
    total_cost = electricity_cost + power_penalty + comfort_penalty
    fitness = 1.0 / max(0.1, total_cost)
    
    return {
        "fitness": fitness,
        "electricity_cost": electricity_cost,
        "power_penalty": power_penalty,
        "comfort_penalty": comfort_penalty,
        "hourly_power": dict(hourly_power)
    }

def crossover(parent1, parent2, user_constraints, devices):
    """改进的交叉操作 - 按设备类型进行智能交叉"""
    child = {}
    # 1. 定义设备类别及其优先级
    device_categories = [
        {
            "name": "essential",
            "devices": ["冰箱"],
            "crossover_method": "elitism"  # 精英保留
        },
        {
            "name": "meal",
            "devices": ["电饭煲_早餐", "电饭煲_午餐", "电饭煲_晚餐"],
            "crossover_method": "whole"  # 整体交叉
        },
        {
            "name": "bath",
            "devices": ["热水器", "洗衣机"],
            "crossover_method": "whole"  # 整体交叉
        },
        {
            "name": "ac",
            "devices": [k for k in parent1.keys() if k.startswith("空调_")],
            "crossover_method": "independent"  # 独立交叉
        }
    ]
    
    # 2. 按类别处理交叉
    for category in device_categories:
        cat_devices = [d for d in category["devices"] 
                      if (d in parent1 or d in parent2)]
        
        if not cat_devices:
            continue
            
        if category["crossover_method"] == "elitism":
            # 精英保留策略 - 选择较优父代的配置
            fit1 = calculate_fitness(parent1, devices, user_constraints)["fitness"]
            fit2 = calculate_fitness(parent2, devices, user_constraints)["fitness"]
            src = parent1 if fit1 > fit2 else parent2
            
            for dev in cat_devices:
                if dev in src:
                    child[dev] = src[dev].copy()
                    
        elif category["crossover_method"] == "whole":
            # 整体交叉策略 - 整个类别一起继承
            p1_has = any(d in parent1 for d in cat_devices)
            p2_has = any(d in parent2 for d in cat_devices)
            
            if p1_has and p2_has:
                # 计算类别适应度
                def category_fitness(parent):
                    temp_ind = {k: v for k, v in parent.items() 
                               if k in cat_devices}
                    return calculate_fitness(temp_ind, devices, user_constraints)["fitness"]
                
                fit1 = category_fitness(parent1)
                fit2 = category_fitness(parent2)
                
                # 80%概率选择较优的，20%随机选择
                if random.random() < 0.8:
                    src = parent1 if fit1 > fit2 else parent2
                else:
                    src = random.choice([parent1, parent2])
            elif p1_has:
                src = parent1
            elif p2_has:
                src = parent2
            else:
                continue
                
            # 继承该类所有设备
            for dev in cat_devices:
                if dev in src:
                    child[dev] = src[dev].copy()
                    
        elif category["crossover_method"] == "independent":
            # 独立交叉策略 - 每个设备独立选择父代
            for dev in cat_devices:
                if dev in parent1 and dev in parent2:
                    # 计算设备局部适应度
                    def device_fitness(parent, d):
                        temp_ind = {d: parent[d]}
                        return calculate_fitness(temp_ind, devices, user_constraints)["fitness"]
                    
                    fit1 = device_fitness(parent1, dev)
                    fit2 = device_fitness(parent2, dev)
                    
                    # 70%概率选择较优的，30%随机选择
                    if random.random() < 0.7:
                        src = parent1 if fit1 > fit2 else parent2
                    else:
                        src = random.choice([parent1, parent2])
                elif dev in parent1:
                    src = parent1
                elif dev in parent2:
                    src = parent2
                else:
                    continue
                    
                child[dev] = src[dev].copy()
    
    # 3. 确保没有遗漏任何设备
    all_devices = set(parent1.keys()).union(set(parent2.keys()))
    for dev in all_devices:
        if dev not in child:
            src = parent1 if dev in parent1 else parent2
            child[dev] = src[dev].copy()
    
    return child

def mutate(individual, devices, user_constraints, mutation_rate=0.05):
    """变异操作"""
    mutated = individual.copy()
    
    for device in list(mutated.keys()):  # 使用list避免遍历时修改字典
        if random.random() < mutation_rate and not device.startswith("空调_"):  # 空调不参与变异
            dev_type = device.split("_")[0]
            params = devices[dev_type]
            
            # 随机选择新的功率等级
            power_level = random.choice(list(params["power_options"].keys()))
            power = params["power_options"][power_level]
            work_time = params["work_time_options"][power_level]
            
            # 根据设备类型确定时间范围
            if dev_type == "电饭煲":
                if "_" in device:  # 确保是电饭煲_早餐这种格式
                    meal_name = device.split("_")[1]
                    meal_times = {
                        "早餐": user_constraints["breakfast_time"],
                        "午餐": user_constraints["lunch_time"],
                        "晚餐": user_constraints["dinner_time"]
                    }
                    if meal_times[meal_name] is not None:
                        end = meal_times[meal_name] - random.uniform(*MEAL_BUFFER_TIME)
                        start = max(0, end - work_time)
                        mutated[device] = {
                            "start": start,
                            "end": end,
                            "power_level": power_level,
                            "power": power,
                            "duration": int(work_time * 60)
                        }
            
            elif dev_type == "热水器":
                if user_constraints["bath_start_time"] is not None:
                    end = user_constraints["bath_start_time"] - random.uniform(*BATH_BUFFER_TIME)
                    start = max(0, end - work_time)
                    mutated[device] = {
                        "start": start,
                        "end": end,
                        "power_level": power_level,
                        "power": power,
                        "duration": int(work_time * 60)
                    }
            
            elif dev_type == "洗衣机":
                if user_constraints["bath_start_time"] is not None and user_constraints["wake_up_time"] is not None:
                    min_start = (user_constraints["bath_start_time"] + 0.5) % 24
                    max_end = (user_constraints["wake_up_time"] - 1) % 24
                    
                    # 改进的时间生成逻辑
                    if min_start < max_end:
                        # 正常情况
                        start = random.uniform(min_start, max(0, max_end - work_time))
                    else:
                        # 跨天情况
                        if random.random() < 0.5:
                            start = random.uniform(min_start, 24 - work_time)
                        else:
                            start = random.uniform(0, max(0, max_end - work_time))
                    
                    # 确保结束时间不早于开始时间
                    end = (start + work_time) % 24
                    if end < start and end > 0:  # 跨天情况
                        end = 24
                    
                    mutated[device] = {
                        "start": start,
                        "end": end,
                        "power_level": power_level,
                        "power": power,
                        "duration": int(work_time * 60)
                    }

    return mutated

def genetic_algorithm(devices, user_constraints, population_size=50, generations=50):
    """遗传算法主函数"""
    hourly_temps = [temp for _, temp in sorted(temperature_data, key=lambda x: x[0])]
    fuzzy_system = create_fuzzy_system(hourly_temps, user_constraints["user_priority"])
    
    population = []
    for _ in range(population_size):
        individual = create_individual(devices, user_constraints, fuzzy_system)
        population.append(individual)
    
    best_history = []
    
    for gen in range(generations):
        fitness_results = [calculate_fitness(ind, devices, user_constraints) for ind in population]
        
        elite_size = max(2, int(population_size * 0.3))
        elites = sorted(zip(population, fitness_results), 
                       key=lambda x: x[1]["fitness"], reverse=True)[:elite_size]
        
        best_ind, best_fit = elites[0]
        best_history.append(best_fit["fitness"])
        print(f"Generation {gen}: Fitness={best_fit['fitness']:.4f} "
              f"Cost={best_fit['electricity_cost']:.2f}+{best_fit['power_penalty']:.2f} "
              f"Comfort={best_fit['comfort_penalty']:.2f}")
        
        new_population = [ind for ind, _ in elites]
        
        while len(new_population) < population_size:
            parents = random.choices(
                population,
                weights=[res["fitness"] for res in fitness_results],
                k=2
            )
            # 80%概率进行交叉，20%概率直接选择较优父代
            if random.random() < 0.8:
                child = crossover(parents[0], parents[1], user_constraints, devices)
            else:
                fit1 = calculate_fitness(parents[0], devices, user_constraints)["fitness"]
                fit2 = calculate_fitness(parents[1], devices, user_constraints)["fitness"]
                child = parents[0] if fit1 > fit2 else parents[1]
                
            child = mutate(child, devices, user_constraints)
            new_population.append(child)
        
        population = new_population
    
    best_individual = max(population, key=lambda x: calculate_fitness(x, devices, user_constraints)["fitness"])
    best_result = calculate_fitness(best_individual, devices, user_constraints)
    
    return {
        "schedule": best_individual,
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

# 修改主程序部分
if __name__ == "__main__":
    # 1. 获取用户输入
    user_input = get_user_input()
    
    # 2. 生成设备配置
    devices = generate_device_profiles()
    
    # 3. 运行遗传算法（原代码误写为粒子群算法）
    print("\n正在优化调度方案...")
    result = genetic_algorithm(devices, user_input)  # 修改为正确的函数名
    
    # 4. 输出结果
    print_schedule(result["schedule"])
    print("\n统计信息:")
    print(f"总电费: {result['statistics']['electricity_cost']:.2f}元")
    print(f"功率惩罚: {result['statistics']['power_penalty']:.2f}元")
    print(f"峰值功率: {result['statistics']['peak_power']:.0f}W")