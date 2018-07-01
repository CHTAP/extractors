"""Auto-generated file, do not edit by hand. GB metadata"""
from ..phonemetadata import NumberFormat, PhoneNumberDesc, PhoneMetadata

PHONE_METADATA_GB = PhoneMetadata(id='GB', country_code=44, international_prefix='00',
    general_desc=PhoneNumberDesc(national_number_pattern='\\d{7,10}', possible_length=(7, 9, 10), possible_length_local_only=(4, 5, 6, 7, 8)),
    fixed_line=PhoneNumberDesc(national_number_pattern='2(?:0[01378]|3[0189]|4[017]|8[0-46-9]|9[0-2])\\d{7}|1(?:1(?:3[0-48]|[46][0-4]|5[0-26-9]|[78][0-49])|21[0-7]|31[0-8]|[4-69]1\\d)\\d{6}|1(?:2(?:0[024-9]|2[3-9]|3[3-79]|4[1-689]|[58][02-9]|6[0-47-9]|7[013-9]|9\\d)|3(?:0\\d|[25][02-9]|3[02-579]|[468][0-46-9]|7[1-35-79]|9[2-578])|4(?:0[03-9]|[28][02-57-9]|[37]\\d|4[02-69]|5[0-8]|[69][0-79])|5(?:0[1-35-9]|2[024-9]|3[015689]|4[02-9]|5[03-9]|6\\d|7[0-35-9]|8[0-468]|9[0-57-9])|6(?:0[034689]|2[0-35689]|[38][013-9]|4[1-467]|5[0-69]|6[13-9]|7[0-8]|9[0124578])|7(?:0[0246-9]|2\\d|3[0236-8]|4[03-9]|5[0-46-9]|6[013-9]|7[0-35-9]|8[024-9]|9[02-9])|8(?:0[35-9]|2[1-57-9]|3[02-578]|4[0-578]|5[124-9]|6[2-69]|7\\d|8[02-9]|9[02569])|9(?:0[02-589]|2[02-689]|3[1-57-9]|4[2-9]|5[0-579]|6[2-47-9]|7[0-24578]|8\\d|9[2-57]))\\d{6}|1(?:2(?:0(?:46[1-4]|87[2-9])|545[1-79]|76(?:2\\d|3[1-8]|6[1-6])|9(?:7(?:2[0-4]|3[2-5])|8(?:2[2-8]|7[0-47-9]|8[345])))|3(?:638[2-5]|647[23]|8(?:47[04-9]|64[0157-9]))|4(?:044[1-7]|20(?:2[23]|8\\d)|6(?:0(?:30|5[2-57]|6[1-8]|7[2-8])|140)|8(?:052|87[123]))|5(?:24(?:3[2-79]|6\\d)|276\\d|6(?:26[06-9]|686))|6(?:06(?:4\\d|7[4-79])|295[567]|35[34]\\d|47(?:24|61)|59(?:5[08]|6[67]|74)|955[0-4])|7(?:26(?:6[13-9]|7[0-7])|442\\d|50(?:2[0-3]|[3-68]2|76))|8(?:27[56]\\d|37(?:5[2-5]|8[239])|84(?:3[2-58]))|9(?:0(?:0(?:6[1-8]|85)|52\\d)|3583|4(?:66[1-8]|9(?:2[01]|81))|63(?:23|3[1-4])|9561))\\d{3}|176888[2-46-8]\\d{2}|16977[23]\\d{3}', example_number='1212345678', possible_length=(9, 10), possible_length_local_only=(4, 5, 6, 7, 8)),
    mobile=PhoneNumberDesc(national_number_pattern='7(?:[1-3]\\d{3}|4(?:[0-46-9]\\d{2}|5(?:[0-689]\\d|7[0-57-9]))|5(?:0[0-8]|[13-9]\\d|2[0-35-9])\\d|7(?:0(?:0[01]|[1-9]\\d)|[1-7]\\d{2}|8[02-9]\\d|9[0-689]\\d)|8(?:[014-9]\\d|[23][0-8])\\d|9(?:[024-9]\\d{2}|1(?:[02-9]\\d|1[028])|3[0-689]\\d))\\d{5}', example_number='7400123456', possible_length=(10,)),
    toll_free=PhoneNumberDesc(national_number_pattern='80(?:0(?:1111|\\d{6,7})|8\\d{7})|500\\d{6}', example_number='8001234567', possible_length=(7, 9, 10)),
    premium_rate=PhoneNumberDesc(national_number_pattern='(?:87[123]|9(?:[01]\\d|8[2349]))\\d{7}', example_number='9012345678', possible_length=(10,)),
    shared_cost=PhoneNumberDesc(national_number_pattern='8(?:4(?:5464\\d|[2-5]\\d{7})|70\\d{7})', example_number='8431234567', possible_length=(7, 10)),
    personal_number=PhoneNumberDesc(national_number_pattern='70\\d{8}', example_number='7012345678', possible_length=(10,)),
    voip=PhoneNumberDesc(national_number_pattern='56\\d{8}', example_number='5612345678', possible_length=(10,)),
    pager=PhoneNumberDesc(national_number_pattern='76(?:0[012]|2[356]|4[0134]|5[49]|6[0-369]|77|81|9[39])\\d{6}', example_number='7640123456', possible_length=(10,)),
    uan=PhoneNumberDesc(national_number_pattern='(?:3[0347]|55)\\d{8}', example_number='5512345678', possible_length=(10,)),
    national_prefix='0',
    preferred_extn_prefix=' x',
    national_prefix_for_parsing='0',
    number_format=[NumberFormat(pattern='(7\\d{3})(\\d{6})', format='\\1 \\2', leading_digits_pattern=['7(?:[1-57-9]|62)', '7(?:[1-57-9]|624)'], national_prefix_formatting_rule='0\\1'),
        NumberFormat(pattern='(\\d{2})(\\d{4})(\\d{4})', format='\\1 \\2 \\3', leading_digits_pattern=['2|5[56]|7[06]'], national_prefix_formatting_rule='0\\1'),
        NumberFormat(pattern='(\\d{3})(\\d{3})(\\d{4})', format='\\1 \\2 \\3', leading_digits_pattern=['1(?:[02-9]1|1)|3|9[018]'], national_prefix_formatting_rule='0\\1'),
        NumberFormat(pattern='(\\d{5})(\\d{4,5})', format='\\1 \\2', leading_digits_pattern=['1(?:38|5[23]|69|76|94)', '1(?:(?:38|69)7|5(?:24|39)|768|946)', '1(?:3873|5(?:242|39[4-6])|(?:697|768)[347]|9467)'], national_prefix_formatting_rule='0\\1'),
        NumberFormat(pattern='(1\\d{3})(\\d{5,6})', format='\\1 \\2', leading_digits_pattern=['1'], national_prefix_formatting_rule='0\\1'),
        NumberFormat(pattern='(800)(\\d{4})', format='\\1 \\2', leading_digits_pattern=['800', '8001', '80011', '800111', '8001111'], national_prefix_formatting_rule='0\\1'),
        NumberFormat(pattern='(845)(46)(4\\d)', format='\\1 \\2 \\3', leading_digits_pattern=['845', '8454', '84546', '845464'], national_prefix_formatting_rule='0\\1'),
        NumberFormat(pattern='(8\\d{2})(\\d{3})(\\d{4})', format='\\1 \\2 \\3', leading_digits_pattern=['8(?:4[2-5]|7[0-3])'], national_prefix_formatting_rule='0\\1'),
        NumberFormat(pattern='(80\\d)(\\d{3})(\\d{4})', format='\\1 \\2 \\3', leading_digits_pattern=['80'], national_prefix_formatting_rule='0\\1'),
        NumberFormat(pattern='([58]00)(\\d{6})', format='\\1 \\2', leading_digits_pattern=['[58]00'], national_prefix_formatting_rule='0\\1')],
    main_country_for_code=True,
    mobile_number_portable_region=True)
