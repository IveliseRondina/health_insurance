select *
from pa004.users u 
	left join pa004.vehicle v on (u.id=v.id)
	left join pa004.insurance i on (u.id=i.id)