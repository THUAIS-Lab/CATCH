import json
import os
import lark_oapi as lark

from lark_oapi.api.auth.v3 import *
from lark_oapi.api.sheets.v3 import *
from lark_oapi.api.drive.v1 import *
from lark_oapi.core.model.base_request import BaseRequest
from lark_oapi.core.model.base_response import BaseResponse
from openpyxl.utils import get_column_letter


# SDK 使用说明: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/server-side-sdk/python--sdk/preparations-before-development
# 以下示例代码默认根据文档示例值填充，如果存在代码问题，请在 API 调试台填上相关必要参数后再复制代码使用
# 复制该 Demo 后, 需要将 "YOUR_APP_ID", "YOUR_APP_SECRET" 替换为自己应用的 APP_ID, APP_SECRET.
def get_tenant_access_token(app_id=None, app_secret=None):
    """
    Returns:
        return json.loads(response.raw.content)['tenant_access_token']
    """
    # 创建client
    client = lark.Client.builder() \
        .app_id(app_id if app_id is not None else os.environ.get('LARK_APP_ID', None)) \
        .app_secret(app_secret if app_secret is not None else os.environ.get('LARK_APP_SECRET', None)) \
        .log_level(lark.LogLevel.WARNING) \
        .build()

    # 构造请求对象
    request: InternalTenantAccessTokenRequest = InternalTenantAccessTokenRequest.builder() \
        .request_body(InternalTenantAccessTokenRequestBody.builder()
            .app_id(app_id if app_id is not None else os.environ.get('LARK_APP_ID', None))
            .app_secret(app_secret if app_secret is not None else os.environ.get('LARK_APP_SECRET', None))
            .build()) \
        .build()

    # 发起请求
    response: InternalTenantAccessTokenResponse = client.auth.v3.tenant_access_token.internal(request)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.auth.v3.tenant_access_token.internal failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
        return

    return json.loads(response.raw.content)['tenant_access_token']

def create_folder(name, folder_token=None, app_id=None, app_secret=None):
    """
    Returns:
        return response.data
        inspect.getmembers: token, url
    """
    # 创建client
    client = lark.Client.builder() \
        .app_id(app_id if app_id is not None else os.environ.get('LARK_APP_ID', None)) \
        .app_secret(app_secret if app_secret is not None else os.environ.get('LARK_APP_SECRET', None)) \
        .log_level(lark.LogLevel.WARNING) \
        .build()

    # 构造请求对象
    request: CreateFolderFileRequest = CreateFolderFileRequest.builder() \
        .request_body(CreateFolderFileRequestBody.builder()
            .name(name)
            .folder_token(folder_token if folder_token is not None else "")
            .build()) \
        .build()

    # 发起请求
    response: CreateFolderFileResponse = client.drive.v1.file.create_folder(request)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.drive.v1.file.create_folder failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
        return

    # 处理业务结果
    lark.logger.info(lark.JSON.marshal(response.data, indent=4))

    return response.data

def create_spreadsheet(title, folder_token=None, app_id=None, app_secret=None) -> Spreadsheet:
    """
    Returns:
        return response.data.spreadsheet
        inspect.getmembers: title, folder_token, url, spreadsheet_token, without_mount
    """
    # 创建client
    client = lark.Client.builder() \
        .app_id(app_id if app_id is not None else os.environ.get('LARK_APP_ID', None)) \
        .app_secret(app_secret if app_secret is not None else os.environ.get('LARK_APP_SECRET', None)) \
        .log_level(lark.LogLevel.WARNING) \
        .build()

    # 构造请求对象
    request: CreateSpreadsheetRequest = CreateSpreadsheetRequest.builder() \
        .request_body(Spreadsheet.builder()
            .title(title)
            .folder_token(folder_token if folder_token is not None else "")
            .build()) \
        .build()

    # 发起请求
    response: CreateSpreadsheetResponse = client.sheets.v3.spreadsheet.create(request)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.sheets.v3.spreadsheet.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
        return

    return response.data.spreadsheet # inspect.getmembers: title, folder_token, url, spreadsheet_token, without_mount

def auth_full_access(token, repo_type='sheet', app_id=None, app_secret=None, open_id=None):
    """
    Returns:
        return response.data.member.__dict__
        keys(): ['member_type', 'member_id', 'perm', 'perm_type', 'type']
    """
    # 创建client
    client = lark.Client.builder() \
        .app_id(app_id if app_id is not None else os.environ.get('LARK_APP_ID', None)) \
        .app_secret(app_secret if app_secret is not None else os.environ.get('LARK_APP_SECRET', None)) \
        .log_level(lark.LogLevel.WARNING) \
        .build()

    # 构造请求对象
    request: CreatePermissionMemberRequest = CreatePermissionMemberRequest.builder() \
        .token(token) \
        .type(repo_type) \
        .need_notification(True) \
        .request_body(BaseMember.builder()
            .member_type("openid")
            .member_id(open_id if open_id is not None else os.environ.get('LARK_OPEN_ID', None))
            .perm("full_access")
            .perm_type("container")
            .type("user")
            .build()) \
        .build()

    # 发起请求
    response: CreatePermissionMemberResponse = client.drive.v1.permission_member.create(request)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.drive.v1.permission_member.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
        return

    return response.data.member.__dict__

def add_sheet(spreadsheet_token, sheet_title, app_id=None, app_secret=None):
    """
    Returns:
        return json.loads(response.raw.content)['data']['replies'][0]['addSheet']['properties']['sheetId']
        {'code': 0,
        'data': {'replies': [{'addSheet': {'properties': {'sheetId': '11jody',
            'title': 'step-2'}}}]},
        'msg': 'success'}
    """
    # 创建client
    client = lark.Client.builder() \
        .app_id(app_id if app_id is not None else os.environ.get('LARK_APP_ID', None)) \
        .app_secret(app_secret if app_secret is not None else os.environ.get('LARK_APP_SECRET', None)) \
        .log_level(lark.LogLevel.WARNING) \
        .build()

    request: BaseRequest = BaseRequest.builder() \
        .http_method(HttpMethod.POST) \
        .uri('/open-apis/sheets/v2/spreadsheets/:spreadsheet_token/sheets_batch_update') \
        .token_types({AccessTokenType.TENANT}) \
        .paths({'spreadsheet_token': spreadsheet_token}) \
        .body({
            'requests': [{
                'addSheet': {
                    'properties': {
                        'title': sheet_title,
                    }
                }
            }]
        }) \
        .build()

    response: BaseResponse = client.request(request)

    if not response.success():
        lark.logger.error(
            f"add sheet failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
        return

    return json.loads(response.raw.content)['data']['replies'][0]['addSheet']['properties']['sheetId']

def get_sheet_id_by_title(spreadsheet_token, sheet_title, app_id=None, app_secret=None):
    """
    Returns:
        return [sheet_id] if sheet_id exists, otherwise []
    """
    client = lark.Client.builder() \
        .app_id(app_id if app_id is not None else os.environ.get('LARK_APP_ID', None)) \
        .app_secret(app_secret if app_secret is not None else os.environ.get('LARK_APP_SECRET', None)) \
        .log_level(lark.LogLevel.WARNING) \
        .build()

    request: BaseRequest = BaseRequest.builder() \
        .http_method(HttpMethod.GET) \
        .uri('/open-apis/sheets/v3/spreadsheets/:spreadsheet_token/sheets/query') \
        .token_types({AccessTokenType.TENANT}) \
        .paths({'spreadsheet_token': spreadsheet_token}) \
        .build()
    
    response: BaseResponse = client.request(request)
    
    return [
        sheet['sheet_id']
        for sheet in json.loads(response.raw.content)['data']['sheets']
            if sheet['title'] == sheet_title
    ]

def append_rows(spreadsheet_token, sheet_id, rows: list[list[str]], start_col=None, end_col=None, app_id=None, app_secret=None):
    """
    Open-source sanitization: private token removed from the example below.
    Returns:
        return json.loads(response.raw.content)['data']
        {'code': 0,
        'data': {'revision': 4,
        'spreadsheetToken': '',
        'tableRange': '2VfHHi!A1:C2',
        'updates': {'revision': 4,
        'spreadsheetToken': '',
        'updatedCells': 6,
        'updatedColumns': 3,
        'updatedRange': '2VfHHi!A1:C2',
        'updatedRows': 2}},
        'msg': 'success'}
    """
    if len(rows) < 1:
        return

    client = lark.Client.builder() \
        .app_id(app_id if app_id is not None else os.environ.get('LARK_APP_ID', None)) \
        .app_secret(app_secret if app_secret is not None else os.environ.get('LARK_APP_SECRET', None)) \
        .log_level(lark.LogLevel.WARNING) \
        .build()

    if start_col is not None:
        if isinstance(start_col, int):
            start_col = get_column_letter(start_col)
    else:
        start_col = get_column_letter(1)
    if end_col is not None:
        if isinstance(end_col, int):
            end_col = get_column_letter(end_col)
    else:
        end_col = get_column_letter(len(rows[0]))

    assert all(len(row) == len(rows[0]) for row in rows), "All rows must have the same number of columns"

    request: BaseRequest = BaseRequest.builder() \
        .http_method(HttpMethod.POST) \
        .uri('/open-apis/sheets/v2/spreadsheets/:spreadsheetToken/values_append') \
        .token_types({AccessTokenType.TENANT}) \
        .paths({'spreadsheetToken': spreadsheet_token}) \
        .body({
            "valueRange": {
                "range": f"{sheet_id}!{start_col}1:{end_col}{str(len(rows))}",
                "values": rows
            }
        }) \
        .build()

    response: BaseResponse = client.request(request)

    if not response.success():
        if response.code is None or response.code == 90227:
            if response.code is None:
                lark.logger.warning(f"request too large, split into multiple requests, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{response.raw.content}")
            else:
                lark.logger.warning(f"request too large, split into multiple requests, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            ret_1 = append_rows(spreadsheet_token, sheet_id, rows[:(len(rows)//2)], start_col, end_col, app_id, app_secret)
            ret_2 = append_rows(spreadsheet_token, sheet_id, rows[(len(rows)//2):], start_col, end_col, app_id, app_secret)
            return ret_2 if ret_2 is not None else ret_1
        lark.logger.error(
            f"add sheet failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
        return

    return json.loads(response.raw.content)['data']

def update_rows(spreadsheet_token, sheet_id, rows: list[list[str]], start_col=None, end_col=None, app_id=None, app_secret=None):
    """
    Open-source sanitization: private token removed from the example below.
    Returns:
        return json.loads(response.raw.content)['data']
        {
            "code": 0,
            "data": {
                "revision": 84,
                "spreadsheetToken": "",
                "updatedCells": 4,
                "updatedColumns": 2,
                "updatedRange": "1QXD0s!A1:B2",
                "updatedRows": 2
            },
            "msg": "success"
        }
    """
    client = lark.Client.builder() \
        .app_id(app_id if app_id is not None else os.environ.get('LARK_APP_ID', None)) \
        .app_secret(app_secret if app_secret is not None else os.environ.get('LARK_APP_SECRET', None)) \
        .log_level(lark.LogLevel.WARNING) \
        .build()

    if start_col is not None:
        if isinstance(start_col, int):
            start_col = get_column_letter(start_col)
    else:
        start_col = get_column_letter(1)
    if end_col is not None:
        if isinstance(end_col, int):
            end_col = get_column_letter(end_col)
    else:
        end_col = get_column_letter(len(rows[0]))

    assert all(len(row) == len(rows[0]) for row in rows), "All rows must have the same number of columns"

    request: BaseRequest = BaseRequest.builder() \
        .http_method(HttpMethod.PUT) \
        .uri('/open-apis/sheets/v2/spreadsheets/:spreadsheetToken/values') \
        .token_types({AccessTokenType.TENANT}) \
        .paths({'spreadsheetToken': spreadsheet_token}) \
        .body({
            "valueRange": {
                "range": f"{sheet_id}!{start_col}1:{end_col}{str(len(rows))}",
                "values": rows
            }
        }) \
        .build()

    response: BaseResponse = client.request(request)

    if not response.success():
        lark.logger.error(
            f"add sheet failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
        return

    return json.loads(response.raw.content)['data']

def set_row_height(spreadsheet_token, sheet_id, row_height=27, start_index=1, end_index=1024, app_id=None, app_secret=None):
    """
    Returns:
        return json.loads(response.raw.content)
        {
            "code": 0,
            "data": {},
            "msg": "Success"
        }
    """
    client = lark.Client.builder() \
        .app_id(app_id if app_id is not None else os.environ.get('LARK_APP_ID', None)) \
        .app_secret(app_secret if app_secret is not None else os.environ.get('LARK_APP_SECRET', None)) \
        .log_level(lark.LogLevel.WARNING) \
        .build()

    request: BaseRequest = BaseRequest.builder() \
        .http_method(HttpMethod.PUT) \
        .uri('/open-apis/sheets/v2/spreadsheets/:spreadsheetToken/dimension_range') \
        .token_types({AccessTokenType.TENANT}) \
        .paths({'spreadsheetToken': spreadsheet_token}) \
        .body({
            "dimension":{
                "sheetId":sheet_id,
                "majorDimension":"ROWS",
                "startIndex":start_index,
                "endIndex":end_index
            },
            "dimensionProperties":{
                "visible":True,
                "fixedSize":row_height
            }
        }) \
        .build()

    response: BaseResponse = client.request(request)

    if not response.success():
        lark.logger.error(
            f"set row height failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
        return

    return json.loads(response.raw.content)