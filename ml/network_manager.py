import subprocess
from bs4 import BeautifulSoup
import json

def start_network():
    subprocess.run('../networks/start_network.sh', shell=True, check=True)

def start_benchmark():
    subprocess.run('../networks/start_benchmark.sh', shell=True, check=True)

def stop_network():
    subprocess.run('../networks/stop_network.sh', shell=True, check=True)

def transform_caliper_html_to_json(html_file, output_json_file):
    """
        Transform html report file to json format
    """
    try:
        with open(html_file, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
        
        summary = soup.find("div", {"id": "summary"}).text.strip() if soup.find("div", {"id": "summary"}) else "No summary found"
        metrics = {}
        
        table = soup.find("table")
        if table:
            headers = [th.text.strip() for th in table.find_all("th")]
            rows = table.find_all("tr")[1:]
            for row in rows:
                cells = row.find_all("td")
                key = cells[0].text.strip()
                values = {headers[i]: cells[i].text.strip() for i in range(1, len(cells))}
                metrics[key] = values
        
        report_data = {
            "summary": summary,
            "metrics": metrics
        }
        
        with open(output_json_file, "w", encoding="utf-8") as json_file:
            json.dump(report_data, json_file, indent=4)
        
        print(f"JSON report successfully created: {output_json_file}")
    except Exception as e:
        print(f"Error during conversion: {e}")


def calculate_average_tps(json_file):
    try:
        json_data = None
        with open(json_file, "r", encoding="utf-8") as file:
            json_data = json.load(file)
        tps_values = []
        for _, metrics in json_data["metrics"].items():
            tps_value = float(metrics.get("Throughput (TPS)", 0))
            tps_values.append(tps_value)
        
        if not tps_values:
            print("No TPS values found.")
            return None
        
        average_tps = sum(tps_values) / len(tps_values)
        return average_tps

    except Exception as e:
        print(f"Error: {e}")
        return None


def observe_data() -> float:
    transform_caliper_html_to_json("../networks/benchmark_report.html", "../networks/benchmark_report.json")
    average_tps = calculate_average_tps("../networks/benchmark_report.json")
    return average_tps