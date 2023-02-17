import requests



host ='converted-serving-2-env.eba-25pxjjyb.us-east-1.elasticbeanstalk.com'
url = f'http://{host}/predict'



id = 'customer'

prospect_id = {
 "lead_origin": "lead_add_form",
 "lead_source": "reference",
 "do_not_email": "no",
 "do_not_call": "no",
 "last_activity": "sms_sent",
 "country": "india",
 "specialization": "select",
 "how_did_you_hear_about_x_education": "select",
 "what_is_your_current_occupation": "unemployed",
 "what_matters_most_to_you_in_choosing_a_course": "better_career_prospects",
 "search": "no",
 "magazine": "no",
 "newspaper_article": "no",
 "digital_advertisement": "no",
 "through_recommendations": "no",
 "receive_more_updates_about_our_courses": "no",
 "tags": "will_revert_after_reading_the_email",
 "lead_quality": "might_be",
 "update_me_on_supply_chain_content": "no",
 "lead_profile": "select",
 "city": "select",
 "asymmetrique_activity_index": "02.medium",
 "asymmetrique_profile_index": "02.medium",
 "i_agree_to_pay_the_amount_through_cheque": "no",
 "a_free_copy_of_mastering_the_interview": "no",
 "last_notable_activity": "sms_sent",
 "newspaper": "no",
 "x_education_forums": "no",
 "get_updates_on_dm_content": "no",
 "lead_number": 627173,
 "totalvisits": 0.0,
 "total_time_spent_on_website": 0,
 "page_views_per_visit": 0.0,
 "asymmetrique_activity_score": 14.306252489048187,
 "asymmetrique_profile_score": 16.344882516925527

}




response = requests.post(url, json=prospect_id).json()
print(response)




if response['converted'] == True:
    print('sending message to % s' %(id))
else:
    print('not sending message to % s' %(id))


