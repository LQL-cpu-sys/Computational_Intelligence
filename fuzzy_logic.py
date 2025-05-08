import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from data import DEVICE_DATA
def create_fuzzy_system(hourly_temperatures, user_priority):
    """创建模糊逻辑系统"""
    # 温度范围扩展为-10到40℃，以更好处理低温情况
    temperature = ctrl.Antecedent(np.arange(-10, 41, 1), 'temperature')
    price_level = ctrl.Antecedent(np.arange(0, 1.5, 0.01), 'price_level')
    user_preference = ctrl.Antecedent(np.arange(0, 3, 1), 'user_preference')
    
    # 输出变量
    power_level = ctrl.Consequent(np.arange(0, 3, 1), 'power_level')  # 0=低, 1=中, 2=高

    # 温度隶属函数
    temperature['cold'] = fuzz.trapmf(temperature.universe, [-10, -10, 16, 20])  # 低温
    temperature['comfortable'] = fuzz.trapmf(temperature.universe, [18, 20, 24, 26])  # 舒适
    temperature['hot'] = fuzz.trapmf(temperature.universe, [24, 26, 40, 40])  # 高温

    # 电价隶属函数
    price_level['low'] = fuzz.trimf(price_level.universe, [0, 0, 0.5])
    price_level['medium'] = fuzz.trimf(price_level.universe, [0.4, 0.7, 1.0])
    price_level['high'] = fuzz.trimf(price_level.universe, [0.8, 1.2, 1.5])
    
    # 用户偏好
    user_preference.automf(3, names=['save', 'balance', 'comfort'])
    
    # 功率等级
    power_level.automf(3, names=['low', 'medium', 'high'])

    # 规则系统
    rules = [
        # 低温情况
        ctrl.Rule(temperature['cold'] & user_preference['comfort'], power_level['high']),
        ctrl.Rule(temperature['cold'] & user_preference['balance'] & price_level['low'], power_level['high']),
        ctrl.Rule(temperature['cold'] & user_preference['balance'] & price_level['medium'], power_level['medium']),
        ctrl.Rule(temperature['cold'] & user_preference['balance'] & price_level['high'], power_level['low']),
        ctrl.Rule(temperature['cold'] & user_preference['save'], power_level['low']),
        
        # 高温情况
        ctrl.Rule(temperature['hot'] & user_preference['comfort'], power_level['high']),
        ctrl.Rule(temperature['hot'] & user_preference['balance'] & price_level['low'], power_level['high']),
        ctrl.Rule(temperature['hot'] & user_preference['balance'] & price_level['medium'], power_level['medium']),
        ctrl.Rule(temperature['hot'] & user_preference['balance'] & price_level['high'], power_level['low']),
        ctrl.Rule(temperature['hot'] & user_preference['save'], power_level['low']),
        
        # 舒适温度
        ctrl.Rule(temperature['comfortable'], power_level['low'])
    ]

    # 创建控制系统
    power_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(power_ctrl)

def get_power_level(fuzzy_system, device_name, current_time, hourly_temps, electricity_price, user_priority):
    """通过模糊系统获取功率等级"""
    hour_idx = int(current_time) % 24
    fuzzy_system.input['temperature'] = hourly_temps[hour_idx]
    fuzzy_system.input['price_level'] = get_price(current_time, electricity_price)
    fuzzy_system.input['user_preference'] = user_priority
    try:
        fuzzy_system.compute()
        pl_index = round(fuzzy_system.output['power_level'])
        return list(DEVICE_DATA[device_name]["功率等级"].keys())[min(pl_index, 2)]
    except:
        return "低"  # 默认低功率

def get_price(hour: float, electricity_price) -> float:
    """获取指定小时的电价"""
    for start, end, price in electricity_price:
        if start <= hour < end:
            return price
    return electricity_price[-1][2]