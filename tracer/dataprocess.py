# encoding: utf-8
import json
import sys
import os
import pandas
import pandas as pd
import numpy as np
import ast
import re
# import numba
import chardet

#set target operator for tensor parallel
TARGET_OP_PREFIXES = [
    'aten::conv2d',
    'aten::linear',
    'aten::embedding',
    # 'autograd::engine::evaluate_function: EmbeddingBackward0',
    # 'autograd::engine::evaluate_function: AddmmBackward0',
    # 'autograd::engine::evaluate_function: ConvolutionBackward0',
    # 'autograd::engine::evaluate_function: ViewBackward0',
    # 'autograd::engine::evaluate_function: MmBackward0'
]

class Kernellist:
    def __init__(self):
        self.name = ''
        self.id = None
        self.duration = None
        self.starttime = None
        self.endtime = None
        self.correlationid = None
        self.registersperthread = None
        self.sharedmemory = None
        self.blocksperSM = None
        self.warpsperSM = None
        self.grid = ''
        self.block = ''
        self.stream = None


class Operatorlist:
    def __init__(self):
        self.name = ''
        self.id = None
        self.duration = None
        self.starttime = None
        self.endtime = None
        self.correlationid = []
        self.cudaevents = []
        self.inputdims = ''
        self.sequenceid =None


class Eventslist:
    def __init__(self):
        self.cpu = []
        self.cuda = []


def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res


def load_json(json_file):
    if json_file is None:
        print("Error: No json file found.")
        return
    print("Analyzing json file: {}".format(json_file))
    with open(json_file, "r") as f:
        json_trace = json.load(f)

def dataprocessLayerv2():
    path = './data/profiler/'
    files = os.listdir(path)
    fileid = 0
    for file in files:
        print(file)
        fileid += 1
        if file == ".DS_Store":
            continue

        with open(path + file, "r" ) as f:
            json_trace = json.load(f)
        cpuevents = []
        profilerstarttime = 0
        for event in json_trace['traceEvents']:
            if (event.get('cat', '').lower() == 'cpu_op') or (event.get('cat', '').lower() == 'operator') and event.get(
                    'ph', '').lower() == 'x':
                dur = event['dur']
                ts = event['ts']
                te = ts + dur
                popitem = []
                aoperator = Operatorlist()
                aoperator.name = event['name']
                aoperator.duration = dur
                aoperator.starttime = ts
                aoperator.endtime = te
                aoperator.inputdims = event['args'].get('Input Dims', [[]])
                # aoperator.inputdims = event['args']['Input dims']
                # aoperator.inputdims = str(event['args']['Input Dims']).replace(' ','').replace(',',';')
                aoperator.sequenceid=event['args'].get('Sequence number')
                cpuevents.append(aoperator)

                for cpueventsitem in cpuevents:
                    if (te <= cpueventsitem.endtime and ts > cpueventsitem.starttime) or (
                            te < cpueventsitem.endtime and ts >= cpueventsitem.starttime) \
                            or (
                            te == cpueventsitem.endtime and ts == cpueventsitem.starttime and aoperator.name != cpueventsitem.name):
                        popitem.append(aoperator)
                    elif te >= cpueventsitem.endtime and ts < cpueventsitem.starttime:
                        popitem.append(cpueventsitem)

                for item in popitem:
                    if item in cpuevents:
                        cpuevents.remove(item)


            elif (event.get('cat', '').lower() == 'cuda_runtime') or (
                    event.get('cat', '').lower() == 'runtime') and event.get('ph', '').lower() == 'x' and event.get(
                'name', '').lower() == 'cudalaunchkernel':
                dur = event['dur']
                ts = event['ts']
                te = ts + dur
                correlationid = event['args']["correlation"]
                for cpueventsitem in cpuevents:
                    if cpueventsitem.endtime > te and cpueventsitem.starttime < ts:
                        cpueventsitem.correlationid.append(correlationid)
            elif event.get('name', '') == 'Iteration Start: PyTorch Profiler':

                profilerstarttime = event.get('ts')

        for event in json_trace['traceEvents']:
            if event.get('cat', '').lower() == 'kernel' and event.get('ph', '').lower() == 'x':
                correlationid = event['args']["correlation"]
                dur = event['dur']
                ts = event['ts']
                te = ts + dur
                cudaevents = []
                for cpueventsitem in cpuevents:
                    if correlationid in cpueventsitem.correlationid:
                        akernel = Kernellist()
                        akernel.name = event['name']
                        akernel.duration = dur
                        akernel.starttime = ts
                        akernel.endtime = te
                        akernel.correlationid = correlationid
                        akernel.registersperthread = event['args']['registers per thread']
                        akernel.sharedmemory = event['args']['shared memory']
                        akernel.blocksperSM = ''  # event['args']['blocks per SM']
                        akernel.warpsperSM = event['args']['warps per SM']
                        akernel.grid = event['args']['grid']
                        akernel.block = event['args']['block']
                        akernel.stream = event['args']['stream']
                        cudaevents.append(akernel)
                        cpueventsitem.cudaevents.append(cudaevents)
        layerid = 0
        data = []
        cpuevents.sort(key=lambda x: x.starttime)
        for cpueventsitem in cpuevents:
            layerid += 1
            mincuda = 0
            maxcuda = 0
            cudatimenooverlap = 0
            for cudaeventsitem in cpueventsitem.cudaevents:
                for item in cudaeventsitem:
                    if mincuda == 0:
                        mincuda = item.starttime - profilerstarttime
                    elif mincuda > item.starttime - profilerstarttime:
                        mincuda = item.starttime - profilerstarttime
                    if maxcuda < item.endtime - profilerstarttime:
                        maxcuda = item.endtime - profilerstarttime
                    cudatimenooverlap += item.endtime - item.starttime
            cudatime = maxcuda - mincuda

            data.append({
                "modelid": file.replace(".json","").replace("profiler_",""),
                "layerid": layerid,
                "cpueventname": cpueventsitem.name,
                "cudatime": cudatime,
                "cudatimenooverlap": cudatimenooverlap,
                "inputdims": str(cpueventsitem.inputdims).replace(' ', '').replace(',', ';'),
                "sequenceid":cpueventsitem.sequenceid
            })

        # Create DataFrame
        df = pd.DataFrame(data)
        # Save to CSV
        directory = os.path.join('./data/middledata/profiler')
        os.makedirs(directory, exist_ok=True)
        df.to_csv(directory +'/'+file.replace(".json", "")+ '.csv', index=False)


def datamerge():
    dir1 = './data/middledata/graph'
    dir2 = './data/middledata/profiler'

    # Iterate through files in the first directory
    for filename in os.listdir(dir1):
        print(filename)
        if filename == ".DS_Store":
            continue
        path_csv1 = os.path.join(dir1, filename)
        path_csv2 = os.path.join(dir2, filename.replace("graph_","profiler_"))
        print(path_csv1)
        print(path_csv2)

        # Check if the corresponding file exists in the second directory
        if os.path.exists(path_csv2):
            # Read the CSV files into DataFrames
            df1 = pd.read_csv(path_csv1)
            df2 = pd.read_csv(path_csv2)

            # Merge the DataFrames on 'layerid' and 'cpueventname'
            merged_df = pd.merge(df1, df2, on=['layerid', 'cpueventname'], suffixes=('_df1', '_df2'))
            merged_df = merged_df[merged_df['cpueventname'] != 'aten::item']
            # merged_df['stage'] = np.where(merged_df['cpueventname'].str.startswith(('autograd', 'aten::_for')), 'backward', 'forward')
            conditions = [
                merged_df['cpueventname'].str.startswith("autograd"),
                merged_df['cpueventname'].str.startswith("aten::_for")
            ]

            choices = ['backward', 'optimizer']
            merged_df['stage'] = np.select(conditions, choices, default='forward')
            
            # merged_df['tpflag'] = np.where(
            #     merged_df['cpueventname'].apply(
            #         lambda name: any(name.startswith(prefix) for prefix in TARGET_OP_PREFIXES)
            #     ),
            #     1, 0
            # )
            # Step 1: Find sids where name is either 'conv' or 'test'
            target_sids = merged_df.loc[merged_df['cpueventname'].isin(TARGET_OP_PREFIXES), 'sequenceid'].unique()
            # Step 2: Initialize flag to 1 for all rows
            merged_df['tpflag'] = 0
            # Step 3: Set flag to 0 for rows with sid in target_sids
            merged_df.loc[merged_df['sequenceid'].isin(target_sids), 'tpflag'] = 1
            # Save the merged DataFrame to a new CSV file if needed
            directory = os.path.join('./data/middledata/mergedata')
            os.makedirs(directory, exist_ok=True)
            merged_df.to_csv(directory +'/'+filename.replace("graph_",""), index=False)

def dataformatfortrace():
    def extract_items(data):
        extracted_items = []
        if isinstance(data[0], list):
            # If the first element is a list of lists, extend the extracted items with its elements
            for sublist in data:
                extracted_items.extend(sublist)
        else:
            # If the first element is a list of integers, add it directly
            extracted_items = data
        return extracted_items
    # Define a function to safely evaluate shapes
    def safe_eval(shape_str):
        try:
            return ast.literal_eval(shape_str)
        except (ValueError, SyntaxError):
            return shape_str

    def parse_op_schema(op_schema_str):
        split_by_commasep = op_schema_str.split(';')
        results = []
        for i, item in enumerate(split_by_commasep):
            if i == 0:
                item = item.split('(')[1]
            elif i == len(split_by_commasep) - 1:
                item = item.split(')')[0]
            item = item.strip()
            item_result = item.split(" ")
            results.append(item_result)
        return results

    def parse_complex_datatype(datatype_str):
        datatype_str = datatype_str.replace('GenericList[', '').replace(']', '')
        items = datatype_str.split(',')
        return [item.strip() for item in items]

    dir = './data/middledata/mergedata/'
    # Iterate through files in the first directory
    for filename in os.listdir(dir):
        if filename == ".DS_Store":
            continue
        path_csv1 = os.path.join(dir, filename)
        df = pd.read_csv(path_csv1)

        # Initialize a dictionary to hold the tensor ID to shape mapping
        tensor_details = []
        operator_details = []

        # Loop through each row in the DataFrame
        for index, row in df.iterrows():
            input_tensorid = []
            output_tensorid = []
            input_size = []
            output_size = []
            # Extract and reformat the strings into valid Python list strings by replacing ';' with ','
            input_shapes_str = '[' + row['inputshapes'].replace(';', ',') + ']'
            input_values_str = '[' + row['inputvalues'].replace(';', ',') + ']'
            input_datatype_str = '[' + row['inputtypes'].replace(';', ',') + ']'
            output_shapes_str = '[' + row['outputshapes'].replace(';', ',') + ']'
            output_values_str = '[' + row['outputvalues'].replace(';', ',') + ']'
            output_datatype_str = '[' + row['outputtypes'].replace(';', ',') + ']'

            # Evaluate the string to get the nested list structure
            input_shapes = safe_eval(input_shapes_str)
            input_values = safe_eval(input_values_str)
            input_datatype = safe_eval(input_datatype_str)
            output_shapes = safe_eval(output_shapes_str)
            output_values = safe_eval(output_values_str)
            output_datatype = safe_eval(output_datatype_str)

            # Extract items from the nested structures
            shape = extract_items(input_shapes)
            value = extract_items(input_values)
            datatype = input_datatype[0]

            o_shape = extract_items(output_shapes)
            o_value = extract_items(output_values)
            o_datatype = output_datatype[0]

            if pd.isna(row['op_schema']):
                continue
            op_schema_str = row['op_schema'].replace('Tensor(a!)','Tensor').replace('Tensor(a)','Tensor').replace('Tensor(a -> *) self','Tensor input')
            op_schema_str = op_schema_str.replace('Tensor self', 'Tensor input')
            if op_schema_str.replace(" ", "") == '':
                continue

            op_schema = parse_op_schema(op_schema_str)
            for i in range(len(value)):
                if "tensor" in datatype[i].lower():
                    if "GenericList" not in datatype[i]:
                        tensor_id = value[i][0]
                        tensor_Storgeid = value[i][1]
                        tensor_elementnum = value[i][3]
                        tensor_elementbyte = value[i][4]
                        tensor_shape = shape[i]
                        tensor_cat = op_schema[i][1]
                        tensor_details.append((tensor_id, tensor_shape, tensor_elementnum, tensor_elementbyte,tensor_cat.replace(")","").replace("(",""),tensor_Storgeid))
                        input_tensorid.append(tensor_id)
                        input_size.append(tensor_elementnum)
                    else:
                        datatype_list = parse_complex_datatype(datatype[i])
                        value_list = value[i]
                        shape_list = shape[i]
                        for index, item in enumerate(datatype_list):
                            tensor_id = value_list[index][0]
                            tensor_Storgeid = value_list[index][1]
                            tensor_elementnum = value_list[index][3]
                            tensor_elementbyte = value_list[index][4]
                            tensor_shape = shape_list[index]
                            # tensor_cat = op_schema[index][1]
                            tensor_cat = 'input'
                            tensor_details.append((tensor_id, tensor_shape, tensor_elementnum, tensor_elementbyte,
                                                   tensor_cat, tensor_Storgeid))
                            input_tensorid.append(tensor_id)
                            input_size.append(tensor_elementnum)

            for i in range(len(o_value)):
                if "tensor" in o_datatype[i].lower():
                    if "GenericList" not in o_datatype[i]:
                        tensor_id = o_value[i][0]
                        tensor_Storgeid = o_value[i][1]
                        tensor_elementnum = o_value[i][3]
                        tensor_elementbyte = o_value[i][4]
                        tensor_shape = o_shape[i]
                        tensor_details.append((tensor_id, tensor_shape, tensor_elementnum, tensor_elementbyte,'output',tensor_Storgeid))
                        output_tensorid.append(tensor_id)
                        output_size.append(tensor_elementnum)
                    else:
                        o_datatype_list = parse_complex_datatype(o_datatype[i])
                        o_value_list = o_value[i]
                        o_shape_list = o_shape[i]
                        for index, item in enumerate(o_datatype_list):
                            tensor_id = o_value_list[index][0]
                            tensor_Storgeid = o_value_list[index][1]
                            tensor_elementnum = o_value_list[index][3]
                            tensor_elementbyte = o_value_list[index][4]
                            tensor_shape = o_shape_list[index]
                            tensor_details.append((tensor_id, tensor_shape, tensor_elementnum, tensor_elementbyte,
                                                   'output', tensor_Storgeid))
                            output_tensorid.append(tensor_id)
                            output_size.append(tensor_elementnum)


            operatorid = row['layerid']
            operatorname = row['cpueventname']
            operator_input = str(input_tensorid).replace(',',';')
            operator_output = str(output_tensorid).replace(',',';')
            operator_cudatime = row['cudatime'] #time unit: us
            operator_cudatimenooverlap= row['cudatimenooverlap'] #single stream use this
            operator_inputsize = str(input_size).replace(',',';')
            operator_outputsize = str(output_size).replace(',', ';')
            operator_gpuid = "0"
            operator_stage = row['stage']
            operator_tpflag = row['tpflag']


            operator_details.append((operatorid,operatorname,operator_input,operator_output,operator_cudatime,
                                     operator_cudatimenooverlap,operator_inputsize,operator_outputsize,
                                     operator_gpuid,operator_stage,operator_tpflag))

        tensor_df = pd.DataFrame(tensor_details, columns=['TensorID', 'TensorShape', 'TensorNumElement', 'TensorEachByte','TensorType','TensorStorgeid'])
        # tensor_df = tensor_df.drop_duplicates(subset=['TensorID']) # remove the duplicate? but the tensor mybe different type in different layer
        tensor_df['Index'] = range(len(tensor_df))
        tensor_df['gpuid'] = "0"
        tensor_df = tensor_df[['Index', 'TensorID', 'TensorShape', 'TensorNumElement', 'TensorEachByte', 'TensorType',
             'TensorStorgeid', 'gpuid']]
        filename = filename.replace(".csv", "")
        # Use re.sub to remove '-iter' and everything after it
        filename = re.sub(r'-iter.*', '', filename)
        directory = os.path.join('./data/middledata/trace', filename)
        # Create the directory if it doesn't exist.
        os.makedirs(directory, exist_ok=True)
        tensor_file = os.path.join(directory, 'tensor.csv')
        tensor_df.to_csv(tensor_file, index=False)

        operator_df = pd.DataFrame(operator_details, columns=['OperatorID', 'OperatorName', 'Operator_input', 'Operator_output','Operator_cudatime','Operator_cudatimenooverlap','InputSize','OutputSize','gpuid','stage','tpflag'])
        # operator_df['gpuid'] = "0"
        operator_df = operator_df[operator_df['Operator_cudatime'] != 0]

        output_file = os.path.join(directory, 'trace.csv')
        operator_df.to_csv(output_file, index=False)

def dataprocessgraphobserverv2():
    path = './data/graph/'
    files = os.listdir(path)
    fileid = 0

    for file in files:
        if file == ".DS_Store":
            continue
        fileid += 1
        if fileid >= 100:
            break
        print(file)
        with open(os.path.join(path, file), "r", encoding='utf-8') as f:
            json_trace = json.load(f)
        datalist = json_trace['nodes']

        # Build mappings for ID and control dependencies
        id_to_node = {node['id']: node for node in datalist}
        ctrl_dep_map = {node['id']: node['parent'] for node in datalist}

        # Collect nodes with specific control dependencies
        nodes_with_ctrl_deps_2 = [node for node in datalist if node.get('parent') == 2]
        nodes_main = [node for node in datalist if node.get('parent') == 1]
        backid = None
        for node in nodes_main:
            if node['id'] not in {1, 2}:
                backid = node['id']

        nodes_with_ctrl_deps_back = [node for node in datalist if node.get('parent') == backid]

        # Initialize data storage
        index = 0
        data = []
        Optimizerid = []
        for node in nodes_with_ctrl_deps_2:
            op_schema = node.get('op_schema', '')
            if "Optimizer" in node['name']:
                Optimizerid.append(node['id'])
            inputshapes = str(node['input_shapes']).replace(' ', '').replace(',', ';')
            inputvalues = str(node['inputs']).replace(' ', '').replace(',', ';')
            inputtypes = str(node['input_types']).replace(' ', '').replace(',', ';')
            outputshapes = str(node['output_shapes']).replace(' ', '').replace(',', ';')
            outputvalues = str(node['outputs']).replace(' ', '').replace(',', ';')
            outputtypes = str(node['output_types']).replace(' ', '').replace(',', ';')

            if not (inputshapes == '[]' and inputvalues=='[]' and inputtypes == '[]' and outputshapes == '[]'
                    and outputvalues=='[]' and outputtypes == '[]'):
                index += 1
                data.append({
                    "modelid": file.replace("graph_", "").replace(".json", ""),
                    "layerid": index,
                    "cpueventname": node['name'],
                    "inputshapes": inputshapes,
                    "inputvalues":inputvalues,
                    "inputtypes": inputtypes,
                    "outputshapes": outputshapes,
                    "outputvalues":outputvalues,
                    "outputtypes": outputtypes,
                    "op_schema": op_schema.replace(',', ';')
                })

        def get_children(node, datalist):
            children = []
            for n in datalist:
                parent = n.get('parent')  # Use 'parent' instead of 'ctrl_deps' in the new format
                if parent == node['id']:
                    children.append(n)
            return children

        for node in nodes_with_ctrl_deps_back:
            children = get_children(node, datalist)
            grandchildren = get_children(children[0], datalist) if children else []
            if grandchildren:
                grandchildren = grandchildren[0]
            else:
                grandchildren = children[0] if children else None
            if not grandchildren:
                continue

            op_schema = grandchildren.get('op_schema', '')

            inputshapes = str(grandchildren.get('input_shapes', [])).replace(' ', '').replace(',', ';')
            inputvalues = str(grandchildren.get('inputs', [])).replace(' ', '').replace(',', ';')
            inputtypes = str(grandchildren.get('input_types', [])).replace(' ', '').replace(',', ';')
            outputshapes = str(grandchildren.get('output_shapes', [])).replace(' ', '').replace(',', ';')
            outputvalues = str(grandchildren.get('outputs', [])).replace(' ', '').replace(',', ';')
            outputtypes = str(grandchildren.get('output_types', [])).replace(' ', '').replace(',', ';')

            if not (inputshapes == '[]' and inputvalues == '[]'and inputtypes == '[]' and outputshapes == '[]'
                    and outputvalues=='[]' and outputtypes == '[]'):
                index += 1
                data.append({
                    "modelid": file.replace("graph_", "").replace(".json", ""),
                    "layerid": index,
                    "cpueventname": node['name'],
                    "inputshapes": inputshapes,
                    "inputvalues":inputvalues,
                    "inputtypes": inputtypes,
                    "outputshapes": outputshapes,
                    "outputvalues":outputvalues,
                    "outputtypes": outputtypes,
                    "op_schema": op_schema.replace(',', ';')
                })

        for opid in Optimizerid:
            opnodes = [node for node in datalist if node.get('parent') == opid]  # Use 'parent' instead of 'ctrl_deps'
            for opsubnode in opnodes:
                op_schema = opsubnode.get('op_schema', '')

                inputshapes = str(opsubnode.get('input_shapes', [])).replace(' ', '').replace(',', ';')
                inputvalues = str(opsubnode.get('inputs', [])).replace(' ', '').replace(',', ';')
                inputtypes = str(opsubnode.get('input_types', [])).replace(' ', '').replace(',', ';')
                outputshapes = str(opsubnode.get('output_shapes', [])).replace(' ', '').replace(',', ';')
                outputvalues = str(opsubnode.get('outputs', [])).replace(' ', '').replace(',', ';')
                outputtypes = str(opsubnode.get('output_types', [])).replace(' ', '').replace(',', ';')

                index += 1
                data.append({
                    "modelid": file.replace("graph_", "").replace(".json", ""),
                    "layerid": index,
                    "cpueventname": opsubnode['name'],
                    "inputshapes": inputshapes,
                    "inputvalues": inputvalues,
                    "inputtypes": inputtypes,
                    "outputshapes": outputshapes,
                    "outputvalues": outputvalues,
                    "outputtypes": outputtypes,
                    "op_schema": op_schema.replace(',', ';')
                })

                # Save to CSV
        df = pd.DataFrame(data)
        directory = os.path.join('./data/middledata/graph')
        os.makedirs(directory, exist_ok=True)
        df.to_csv(directory +'/'+file.replace(".json", "")+ '.csv', index=False)

if __name__ == "__main__":
    dataprocessLayerv2() #map forward and backward operator
    dataprocessgraphobserverv2() #for "schema": "1.0.1", version of graph observer
    datamerge()
    dataformatfortrace()
    print("trace is under ./data/middledata/trace")

