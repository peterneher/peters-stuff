import json
import datetime
import xlsxwriter

# google history json file
with open('GoogleLocationHistory.json') as data_file:
    data = json.load(data_file)

locations = data['locations']

worktime = 0
last_date = None
last_work_date_end = None
last_work_date_begin = None

last_month = None
check_dates = []

workbook = xlsxwriter.Workbook('Timesheet.xlsx')
bold = workbook.add_format({'bold': True})
worksheet = None
row = 0
sum = 0

for loc in locations:
    time = int(loc['timestampMs'])
    latitude = int(loc['latitudeE7']) / 100000
    longitute = int(loc['longitudeE7']) / 100000

    date = datetime.datetime.fromtimestamp(time / 1e3)
    if date.year < 2016 or date.month < 10 :
        continue

    if last_date is not None:
        timediff = (last_date - date).total_seconds()

		# replace with location coordinates of your choice
        if (latitude == 4942 or latitude == 4941) and longitute == 867 and last_date.day == date.day:

            if timediff > 0:
                worktime += timediff
                last_work_date_begin = date
                if last_work_date_end is None or date.hour > last_work_date_end.hour:
                    last_work_date_end = date

        elif last_date.day != date.day:

            if last_date.strftime("%A")=='Sunday' :
                row += 1

            if worktime > 0:

                wt = worktime / 3600
                date_str = str(last_work_date_end.day) + '.' + str(last_work_date_end.month) + '.'
                print(date_str)
                print("worked for " + str(wt) + 'h')

                worksheet.write(row, 0, last_work_date_end.strftime("%A"))
                worksheet.write(row, 1, date_str)
                worksheet.write(row, 2, str(last_work_date_begin.hour) + ':' + str(last_work_date_begin.minute))
                worksheet.write(row, 3, str(last_work_date_end.hour) + ':' + str(last_work_date_end.minute))
                worksheet.write(row, 4, round(wt,2))

                if wt < 6 or last_work_date_end.hour > 19:
                    check_dates.append([last_work_date_end, wt])

                    remark = ''
                    if wt < 6 :
                        remark += 'not much work?'
                    elif wt > 10 :
                        remark += 'too much work?'
                    if last_work_date_end.hour > 19 :
                        if len(remark)>0 :
                            remark += '\nworked late?'
                        else:
                            remark += 'worked late?'
                    worksheet.write(row, 5, remark)
            else :
                worksheet.write(row, 0, last_date.strftime("%A"))
                date_str = str(last_date.day) + '.' + str(last_date.month) + '.'
                worksheet.write(row, 1, date_str)

            row += 1


            worktime = 0
            last_work_date_end = None
            last_work_date_begin = None

    last_date = date

    if date.month != last_month :
        if last_month is not None :
            worksheet.write(row, 4, '=SUMME(E1:E'+str(row)+')')

        last_month = date.month
        worksheet = workbook.add_worksheet(name=str(date.month) + '.' + str(date.year))
        worksheet.write(0, 0, 'Day', bold)
        worksheet.write(0, 1, 'Date', bold)
        worksheet.write(0, 2, 'Start', bold)
        worksheet.write(0, 3, 'End', bold)
        worksheet.write(0, 4, 'Work time', bold)
        worksheet.write(0, 5, 'Remarks', bold)
        row = 1
        sum = 0

print('\nCheck dates:')
for d in check_dates:
    print('\n')
    print(d[0])
    # print(str(d[0].day) + '.' + str(d[0].month) + '.' + str(d[0].year))
    print("worked for " + str(d[1]) + 'h')

workbook.close()

